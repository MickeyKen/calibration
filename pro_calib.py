import numpy as np
import cv2 as cv
import glob

import matplotlib
import matplotlib.pyplot as plt

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('*.jpeg')

for fname in images:

    img = cv.imread(fname)
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

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (9,7), None)

    ret2, circles = cv.findCirclesGrid(gray_for_circle, (10,7), flags = cv.CALIB_CB_SYMMETRIC_GRID)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(new_img, (9,7), corners2, ret)

    if ret2 == True:
        objpoints.append(objp)
        imgpoints.append(circles)
        cv.drawChessboardCorners(new_img, (10, 7), circles, ret2)

    cv.imshow('img', new_img)
    #cv.imshow('inv_img',invgray)
    cv.waitKey(1000)

cv.destroyAllWindows()
