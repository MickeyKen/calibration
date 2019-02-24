import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

img = cv.imread("1.jpeg")
new_img = img.copy()
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
invgray = cv.bitwise_not(gray)

orgHeight, orgWidth = img.shape[:2]
for i in range(orgHeight):
  for j in range(orgWidth):
    b = img[i][j][0]
    g = img[i][j][1]
    r = img[i][j][2] 
    if b > 230 and g > 230 and r > 230:
      img[i][j][0] = 0
      img[i][j][1] = 0 
      img[i][j][2] = 0 

gray_for_circle = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 

ret2, circles = cv.findCirclesGrid(gray_for_circle, (10,7), flags = cv.CALIB_CB_SYMMETRIC_GRID)

if ret2 == True:
  objpoints.append(objp)
  imgpoints.append(circles)
  cv.drawChessboardCorners(new_img, (10, 7), circles, ret2)
  print circles[69][0]


#cv.imshow('img', new_img)
#cv.imshow('inv_img',invgray)
#cv.waitKey(1000)

#cv.destroyAllWindows()
