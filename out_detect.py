import cv2
import numpy as np

row = 9
col = 7
corner_num = (col, row)
size = 2.5

pW = np.empty([row * col, 3], dtype=np.float32)
K = np.array([[ 1.0464088296606685e+03, 0., 9.6962285013582118e+02], [0.,
       1.0473601981442353e+03, 5.3418043955010319e+02], [0., 0., 1. ]],np.float32)
d = np.array([[ 4.3977277514639868e-02, -6.2933078892199332e-02,
       -5.7377837329916246e-04, 7.8218303056817190e-04,
       1.4687866930870116e-02 ]],np.float32)

for i_row in range(0, row):
  for i_col in range(0, col):
    pW[i_row * col + i_col] = np.array([size * i_col, size * i_row, 0], dtype=np.float32)

ex_image = cv2.imread("1.jpeg", 0)
found, qI = cv2.findChessboardCorners(ex_image, corner_num)

if found:
  term = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_COUNT, 30, 0.1)
  qI_sub = cv2.cornerSubPix(ex_image, qI, (9,9), (-1,-1), term)

  ret, rvec, tvec = cv2.solvePnP(pW, qI, K, d)
  rmat = cv2.Rodrigues(rvec)[0]
  print rmat
