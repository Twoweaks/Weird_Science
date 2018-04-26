#import libraries
import os
import cv2
import numpy as np
import math
import sys
import dlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()


img_folder_path = 'images/'
imgPaths = [img_folder_path+name for name in os.listdir(img_folder_path) if not name[0] == '.']
imgNames = [name for name in os.listdir(img_folder_path) if not name[0] == '.']

for index,img in enumerate(imgPaths):
    img = cv2.imread(img)
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        shape = sp(img, d)
    
    filename = str(img_folder_path) + str(imgNames[index]) + '.txt'
    with open(filename, 'w') as fh:
        for i in range(shape.num_parts):
            fh.write('{} {}\n'.format(shape.part(i).x, shape.part(i).y))        


def readPoints(path) :
    pointsArray = [];

 #List all files in the directory and read points from text files one by one
    for filePath in os.listdir(path):
        
        if filePath.endswith(".txt"):
            
            #Create an array of points.
            points = [];            
            
            # Read points from filePath
            with open(os.path.join(path, filePath)) as file :
                for line in file :
                    x, y = line.split()
                    points.append((int(x), int(y)))
            
            # Store array of points
            pointsArray.append(points)
            
    return pointsArray;

# Read all jpg images in folder.
def readImages(path) :
    
    #Create array of array of images.
    imagesArray = [];
    
    #List all files in the directory and read points from text files one by one
    for filePath in os.listdir(path):
        if filePath.endswith(".jpg"):
            # Read image found.
            img = cv2.imread(os.path.join(path,filePath));

            # Convert to floating point
            img = np.float32(img)/255.0;

            # Add to array of images
            imagesArray.append(img);
            
    return imagesArray;
                
def similarityTransform(inPoints, outPoints) :
    s60 = math.sin(60*math.pi/180);
    c60 = math.cos(60*math.pi/180);  
  
    inPts = np.copy(inPoints).tolist();
    outPts = np.copy(outPoints).tolist();
    
    xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0];
    yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1];
    
    inPts.append([np.int(xin), np.int(yin)]);
    
    xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0];
    yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1];
    
    outPts.append([np.int(xout), np.int(yout)]);
    
    tform = cv2.estimateRigidTransform(np.array([inPts]), np.array([outPts]), False);
    
    return tform;


# Check if a point is inside a rectangle
def rectContains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True



# Calculate delanauy triangle
def calculateDelaunayTriangles(rect, points):
    # Create subdiv
    subdiv = cv2.Subdiv2D(rect);
   
    # Insert points into subdiv
    for p in points:
        subdiv.insert((p[0], p[1]));

   
    # List of triangles. Each triangle is a list of 3 points ( 6 numbers )
    triangleList = subdiv.getTriangleList();

    # Find the indices of triangles in the points array

    delaunayTri = []
    
    for t in triangleList:
        pt = []
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))
        
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])        
        
        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            for j in range(0, 3):
                for k in range(0, len(points)):                    
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)                            
            if len(ind) == 3:                                                
                delaunayTri.append((ind[0], ind[1], ind[2]))
        

    
    return delaunayTri


def constrainPoint(p, w, h) :
    p =  ( min( max( p[0], 0 ) , w - 1 ) , min( max( p[1], 0 ) , h - 1 ) )
    return p;


def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = [] 
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    
    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    
    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
     
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect
    
    
w = 500;
h = 500;


# Read points for all images
allPoints = readPoints(img_folder_path);

# Read all images
images = readImages(img_folder_path);

# Eye corners
eyecornerDst = [ (np.int(0.3 * w ), np.int(h / 3)), (np.int(0.7 * w ), np.int(h / 3)) ];

imagesNorm = [];
pointsNorm = [];

# Add boundary points for delaunay triangulation
boundaryPts = np.array([(0,0), (w/2,0), (w-1,0), (w-1,h/2), ( w-1, h-1 ), ( w/2, h-1 ), (0, h-1), (0,h/2) ]);

# Initialize location of average points to 0s
pointsAvg = np.array([(0,0)]* ( len(allPoints[0]) + len(boundaryPts) ), np.float32());

n = len(allPoints[0]);

numImages = len(images)

# Warp images and trasnform landmarks to output coordinate system,
# and find average of transformed landmarks.

for i in range(0, numImages):

    points1 = allPoints[i];

    # Corners of the eye in input image
    eyecornerSrc  = [ allPoints[i][36], allPoints[i][45] ] ;
    
    # Compute similarity transform
    tform = similarityTransform(eyecornerSrc, eyecornerDst);
    
    # Apply similarity transformation
    img = cv2.warpAffine(images[i], tform, (w,h));

    # Apply similarity transform on points
    points2 = np.reshape(np.array(points1), (68,1,2));        
    
    points = cv2.transform(points2, tform);
    
    points = np.float32(np.reshape(points, (68, 2)));
    
    # Append boundary points. Will be used in Delaunay Triangulation
    points = np.append(points, boundaryPts, axis=0)
    
    # Calculate location of average landmark points.
    pointsAvg = pointsAvg + points / numImages;
    
    pointsNorm.append(points);
    imagesNorm.append(img);



# Delaunay triangulation
rect = (0, 0, w, h);
dt = calculateDelaunayTriangles(rect, np.array(pointsAvg));

# Output image
output = np.zeros((h,w,3), np.float32());
IMAGES = []
# Warp input images to average image landmarks
for i in range(0, len(imagesNorm)) :
    img = np.zeros((h,w,3), np.float32());
    IMAGES.append(img)
    # Transform triangles one by one
    for j in range(0, len(dt)) :
        tin = []; 
        tout = [];
        
        for k in range(0, 3) :                
            pIn = pointsNorm[i][dt[j][k]];
            pIn = constrainPoint(pIn, w, h);
            
            pOut = pointsAvg[dt[j][k]];
            pOut = constrainPoint(pOut, w, h);
            
            tin.append(pIn);
            tout.append(pOut);
        
        
        warpTriangle(imagesNorm[i], img, tin, tout);  


X_train = IMAGES[:-2]   
X_test = IMAGES[-2:]   
# Add photos (last2 - examples to compare)
for i in range(0, len(X_train)) :        
# Add image intensities for averaging
    output = output + IMAGES[i];

# Divide by numImages to get averagex, y = line.split()
output = output / len(X_train);
output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)


plt.imshow(output)
plt.savefig(str(img_folder_path) +'LIZA.png')

X_train_flatten = [np.reshape(x, (w*h*3)) for x in np.vstack((X_train,X_test))]

pca = PCA(n_components= 500,  svd_solver='randomized', random_state = 13)
pca.fit(X_train_flatten)
X = pca.transform(X_train_flatten)

db = DBSCAN(eps=0.5, min_samples=5).fit(X[:-2])
X_pd = pd.DataFrame(X[:-2])
X_pd['labels'] = db.labels_

X_like_cluster =X_pd[X_pd['labels'] == X_pd['labels'].value_counts().idxmax()].drop('labels',axis = 1)

dist_1 = 0
dist_2 = 0
for i in range(X_like_cluster.shape[0]):
    dist_1+= np.linalg.norm(X[11]-X_like_cluster.iloc[i])
    dist_2+= np.linalg.norm(X[12]-X_like_cluster.iloc[i])

scaler = MinMaxScaler()
weights =scaler.fit_transform(np.array([dist_1,dist_2]).reshape(-1,1))
# weights
