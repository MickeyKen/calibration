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

objectPointsAccum = []
projCirclePoints = []


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
   
        #get homography
        pts_src = np.array([[0.0,0.0], [1919.0,0.0], [1919.0,1079.0],[0.0,1079.0]],np.float32)
        pts_dst = np.array([[corners2[0][0][0],corners2[0][0][1]], [corners2[8][0][0],corners2[8][0][1]], [corners2[62][0][0],corners2[62][0][1]], [corners2[54][0][0],corners2[54][0][1]]],np.float32)

        h, status = cv.findHomography(pts_src, pts_dst)


    if ret2 == True:
        objpoints.append(objp)
        imgpoints.append(circles)
        #print circles[0][0]
        #print circles[9][0]
        #print circles[69][0]
        #print circles[60][0]
        objectPointsAccum.push()
        projCirclePoints.push(circles)

    #cv.imshow('img', new_img)
    #cv.imshow('inv_img',invgray)
    #cv.waitKey(1000)

#cv.destroyAllWindows()
ret, K_proj, dist_coef_proj, rvecs, tvecs = cv.calibrateCamera(objectPointsAccum,
                                                                projCirclePoints,
                                                                (w_proj, h_proj),
                                                                K_proj,
                                                                dist_coef_proj,
                                                                flags = cv.CALIB_USE_INTRINSIC_GUESS)
print("proj calib mat after\n%s"%K_proj)
print("proj dist_coef %s"%dist_coef_proj.T)
print("calibration reproj err %s"%ret)
