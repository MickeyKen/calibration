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
print img.shape[:2]
#new_img = img.copy()
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
#invgray = cv.bitwise_not(gray)

ret, corners = cv.findChessboardCorners(gray, (9,7), None)

# If found, add object points, image points (after refining them)
if ret == True:
  objpoints.append(objp)
  corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
  imgpoints.append(corners)
  # Draw and display the corners
  cv.drawChessboardCorners(img, (9,7), corners2, ret)
  #print len(corners2) 
  #max value = print corners2[62][0]
  #print type(corners2)
  pts_src = np.array([[0.0,0.0], [1919.0,0.0], [1919.0,1079.0],[0.0,1079.0]],np.float32)
  pts_dst = np.array([[corners2[0][0][0],corners2[0][0][1]], [corners2[8][0][0],corners2[8][0][1]], [corners2[62][0][0],corners2[62][0][1]], [corners2[54][0][0],corners2[54][0][1]]],np.float32)

  #print pts_src
  #print pts_dst

  h, status = cv.findHomography(pts_src, pts_dst)

  #warp = cv.warpPerspective(img, h, (1920,1080))

  

cv.imshow('img', warp)
cv.waitKey(0)

cv.destroyAllWindows()
