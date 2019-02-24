import argparse
import cv2
import numpy as np

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
def get_args():
    parser = argparse.ArgumentParser(description="This script creates a circleboard image for calibration")
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=7)
    parser.add_argument("--margin_size", type=int, default=12)
    parser.add_argument("--block_size", type=int, default=100)
    parser.add_argument("--radius", type=int, default=30)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    w = args.width
    h = args.height
    margin = args.margin_size
    block_size = args.block_size
    radius = args.radius
    chessboard = np.ones((block_size * h + margin * 2, block_size * w + margin * 2), dtype=np.uint8) * 255

    for y in range(h):
        for x in range(w):
            cx = int((x + 0.5) * block_size + margin)
            cy = int((y + 0.5) * block_size + margin)
            cv2.circle(chessboard, (cx, cy), radius, 0, thickness=-1)

    #cv2.imwrite("circleboard{}x{}.png".format(w, h), chessboard)

    ch = 1024
    cw = 768
    chessboard = cv2.resize(chessboard,(ch,cw))
    #print chessboard.shape[:2]

    ret2, circles = cv2.findCirclesGrid(chessboard, (10,7), flags = cv2.CALIB_CB_SYMMETRIC_GRID)

    if ret2 == True:
        objpoints.append(objp)
        imgpoints.append(circles)
        cv2.drawChessboardCorners(chessboard, (10, 7), circles, ret2)
        print circles[0][0]
        print circles[9][0]
        print circles[69][0]
        print circles[60][0]
        

    #cv2.imshow('img', chessboard)
    #cv.imshow('inv_img',invgray)
    #cv2.waitKey(0)
    
    #cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
