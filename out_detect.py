import cv2
import numpy as np

row = 9
col = 7
corner_num = (col, row)
size = 2.5

pW = np.empty([row * col, 3], dtype=np.float32)

for i_row in range(0, row):
  for i_col in range(0, col):
    pW[i_row * col + i_col] = np.array([size * i_col, size * i_row, 0], dtype=np.float32)

ex_image = cv2.imread("1.jpeg", 0)
found, qI = cv2.findChessboardCorners(ex_image, corner_num)

if found:
  term = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_COUNT, 30, 0.1)
  qI_sub = cv2.cornerSubPix(ex_image, qI, (5,5), (-1,-1), term)

  pts_src = np.array([[0.0,0.0], [1919.0,0.0], [1919.0,1079.0],[0.0,1079.0]],np.float32)
  ret, rvec, tvec = cv2.solvePnP(qW, qI, K, d)
  rmat = cv2.Rodrigues(rvec)[0]
