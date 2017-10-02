import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import sys

def calibrate_camera_cache():
    # return a cached calibration of the twenty test images to speed up model

    mtx = np.array([[  1.15396093e+03,   0.00000000e+00,   6.69705357e+02],
       [  0.00000000e+00,   1.14802496e+03,   3.85656234e+02],
       [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

    dist = np.array([[ -2.41017956e-01,  -5.30721170e-02,  -1.15810353e-03,
         -1.28318860e-04,   2.67125310e-02]])

    return mtx, dist

def calibrate(img_fnames, save_img = False):

    objpoints = [] #3D coordinates in real world
    imgpoints = [] #2D coordinates in image

    objp = np.zeros((9*6,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    for fname in img_fnames:
        img = mpimg.imread(fname)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)

            if save_img:
                img_corner = cv2.drawChessboardCorners(img, (9,6), corners, ret)
                out_fname = fname.split(".")[0]+'_corner.jpg'
                mpimg.imsave(out_fname, img_corner)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if save_img:
        for fname in img_fnames:
            img = mpimg.imread(fname)
            img_undistort = cv2.undistort(img, mtx, dist, None, mtx)
            out_fname = fname.split(".")[0]+'_undistorted.jpg'
            mpimg.imsave(out_fname, img_undistort)

    return mtx, dist

def calibrate_perspective_cache():
    # top_left = [580,480]
    # top_right = [760,480]
    # bottom_right = [1170,720]
    # bottom_left = [250,720]
    #
    # top_left_dst = [250,0]
    # top_right_dst = [1100,0]
    # bottom_right_dst = [1170,720]
    # bottom_left_dst = [250,720]

    top_left = [600,450]
    top_right = [680,450]
    bottom_right = [1110,720]
    bottom_left = [200,720]

    top_left_dst = [300,0]
    top_right_dst = [850,0]
    bottom_right_dst = [850,720]
    bottom_left_dst = [300,720]


    src_pts = np.array([bottom_left,bottom_right,top_right,top_left])
    dst_pts = np.array([bottom_left_dst, bottom_right_dst, top_right_dst, top_left_dst])
    #
    # copy = mpimg.imread('test_images/straight_lines1.jpg')
    # cv2.polylines(copy,[src_pts],True,(255,0,0), 5)
    # plt.imshow(copy)
    # plt.show()

    src_pts = np.float32(src_pts.tolist())
    dst_pts = np.float32(dst_pts.tolist())
    return src_pts, dst_pts

if __name__ == '__main__':
    img_fnames = glob.glob('camera_cal/calibration[0-9].jpg') + glob.glob('camera_cal/calibration[0-9][0-9].jpg')
    save_img = sys.argv[1] if len(sys.argv) == 2 else False
    # mtx, dist = calibrate(img_fnames, save_img=save_img)
    # print("mtx: ", mtx)
    # print("dist: ", dist)

    calibrate_perspective_cache()
