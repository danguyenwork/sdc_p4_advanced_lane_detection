from calibration import calibrate_perspective_cache, calibrate_camera_cache
from frame import Frame
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import os
import itertools
import numpy as np
from moviepy.editor import VideoFileClip
from lane_tracker import LaneTracker
import sys

def plot(img_lst):
    f, axarr = plt.subplots(len(img_lst))
    for index, img in enumerate(img_lst):
        if len(img.shape) == 2:
            axarr[index].imshow(img, cmap='gray')
        else:
            axarr[index].imshow(img)
    plt.show()

def color_threshold_experiment(frame):
    # low = np.mgrid[9:18] * 10
    s_low = [100]
    s_high = [250]

    l_low = [0]
    l_high = [120]

    s_threholds = list(itertools.product(s_low, s_high))
    l_threholds = list(itertools.product(l_low, l_high))
    color_masks = {}
    for index, _ in enumerate(s_threholds):
        s_mask = frame.hls_color_thresh(2, s_threholds[index])
        l_mask = frame.hls_color_thresh(1, l_threholds[index])
        color_masks[index] = np.zeros_like(s_mask)
        color_masks[index][(s_mask == 1) & (l_mask == 0)] = 1
        # plot([s_mask, l_mask, color_masks[index]])
        # import ipdb; ipdb.set_trace()

    return color_masks

def sobel_threshold_one(frame, args):
    ksize = 5

    grad_thresh, mag_thresh, dir_thresh = args
    gradx = frame.abs_sobel_thresh(orient='x', sobel_kernel=ksize, thresh=grad_thresh)
    grady = frame.abs_sobel_thresh(orient='y', sobel_kernel=ksize, thresh=grad_thresh)
    mag_binary = frame.mag_thresh(sobel_kernel=ksize, thresh=mag_thresh)
    dir_binary = frame.dir_threshold(sobel_kernel=ksize, thresh=dir_thresh)

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined

def sobel_threshold_experiment(frame):
    grad_thresh = [(30,90)]
    mag_thresh = [(30,90)]
    dir_thresh = [(0.8,1.2)]

    thresholds = list(zip(itertools.product(grad_thresh, mag_thresh, dir_thresh)))

    sobel_mask = {}
    for index, thresh in enumerate(thresholds):
        sobel_mask[index] = sobel_threshold_one(frame, thresh[0])

    return sobel_mask

def pipeline(img):
    frame = Frame(img)

    calibrate_params = calibrate_camera_cache()
    perspective_params = calibrate_perspective_cache()

    frame.set_undistort_params(calibrate_params)
    frame.undistorted = frame.undistort(frame.rgb)

    frame.gray = frame.gray(frame.undistorted)
    frame.hls = frame.hls(frame.undistorted)
    frame.hsv = frame.hsv(frame.undistorted)

    frame.color_masks = color_threshold_experiment(frame)[0]
    frame.sobel_masks = sobel_threshold_experiment(frame)[0]

    frame.color_binary = np.dstack(( np.zeros_like(frame.sobel_masks), frame.sobel_masks, frame.color_masks))

    frame.combined_binary = np.zeros_like(frame.color_masks)

    frame.combined_binary[(frame.color_masks == 1) | (frame.sobel_masks == 1)] = 1

    # frame.region_of_interest = frame.region_of_interest(frame.combined_binary)

    frame.set_perspective_params(perspective_params)

    frame.perspective_transformed = frame.perspective_transform(frame.combined_binary)

    lane_tracker.append_frame(frame)

    frame.lane = frame.draw_lane_line(lane_tracker)

    frame.unwarped_lane = frame.draw_unwarped_lane()

    if write_images:
        frame.write_images()

    # import ipdb; ipdb.set_trace()

    return frame.unwarped_lane

lane_tracker = LaneTracker()
write_images = False

if __name__ == '__main__':
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    write_images = (sys.argv[3] == 'True')
    imgmode = (sys.argv[4] == 'img')

    if imgmode:
        test_fnames = glob.glob('test_images/*.jpg')
        for fname in test_fnames:
            img = mpimg.imread(fname)
            pipeline(img)
    else:
        test_videos = glob.glob('test_videos/project_video.mp4')
        for fname in test_videos:
            if end != 0:
                inp = VideoFileClip(fname).subclip(start,end)
            else:
                inp = VideoFileClip(fname)
            output = inp.fl_image(pipeline)
            output_dir = fname.split(".")[0] + '_output_' + str(start) + '_' + str(end) + '_.mp4'
            output.write_videofile(output_dir, audio=False)
