import cv2
import numpy as np
from glob import glob 

row = 7
col = 9
corner_num = (row, col)
size = 5

pW = np.empty([row *col, 3], dtype=np.float32)

for i_row in range(0, row):
  for i_col in range(0, col):
    pW[i_row * col + i_col] = np.array([size * i_col, size * i_row, 0], dtype = np.float32)

pWs = []
qIs = []

for path_image in glob("*.jpeg"):
  image = cv2.imread(path_image, 0)
  found, qI = cv2.findChessboardCorners(image, corner_num)

  if found:
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
    qI_sub = cv2.cornerSubPix(image, qI, (7,7), (-1,-1), term)

    pWs.append(pW)
    print pW
    qIs.append(qI_sub)

  else:
    continue

rep, K, d, rvec, tvec = cv2.calibrateCamera(pWs, qIs, (image.shape[1], image.shape[0]), None, None)
