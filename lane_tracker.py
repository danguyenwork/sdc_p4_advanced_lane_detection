import numpy as np
from lane import Lane

class LaneTracker(object):
    def __init__(self, margin = 100, minpix = 50):
        # self.frames = []
        self.current_frame = None
        self.last_frame = None
        self.margin = margin
        self.minpix = minpix
        self.last_valid_left_lane_found = 0
        self.last_valid_right_lane_found = 0

    def append_frame(self, frame):
        # self.frames.append(frame)
        self.last_frame = self.current_frame
        self.current_frame = frame
        self.img = self.current_frame.perspective_transformed

    def _initialize_sliding_window_params(self):
        # the starting x-coordinates of the sliding windows, leftx_base and rightx_base
        # determined by finding the two largest x_coordinates of the histogram of the bottom half of the warped image
        img = self.img
        histogram = np.sum(img[img.shape[0] // 2:,:], axis=0)
        midpoint = histogram.shape[0] // 2

        params = {}

        params['leftx_base'] = np.argmax(histogram[:midpoint])
        params['rightx_base'] = np.argmax(histogram[midpoint:]) + midpoint

        # set the number of windows and corresponding window height
        params['nwindows'] = 9
        params['window_height'] = int(img.shape[0] / params['nwindows'])

        # nonzero: the array of nonzero value pixels (i.e. white pixels)
        # nonzerox: the array of white pixels in the x coordinate
        # nonzeroy: the array of white pixels in the y coordinate
        params['nonzero'] = img.nonzero()
        params['nonzerox'] = np.array(params['nonzero'][1])
        params['nonzeroy'] = np.array(params['nonzero'][0])

        # initialize the current x-coordinate of the window to the base for the first window
        params['leftx_current'] = params['leftx_base']
        params['rightx_current'] = params['rightx_base']

        # set how many pixels need to be present for the next window to be recentered
        # on the leftx_current and rightx_current of the current windows

        params['left_lane_inds'] = []
        params['right_lane_inds'] = []

        return params

    def _sliding_window_one(self, window, params):
        img = self.img
        margin = self.margin
        minpix = self.minpix

        window_height = params['window_height']

        leftx_current = params['leftx_current']
        rightx_current = params['rightx_current']
        nonzeroy = params['nonzeroy']
        nonzerox = params['nonzerox']
        left_lane_inds = params['left_lane_inds']
        right_lane_inds = params['right_lane_inds']

        # determine the coordinates of the two search boxes for the current window
        win_y_low = img.shape[0] - (window+1) * window_height
        win_y_high = img.shape[0] - (window) * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        return leftx_current, rightx_current

    def _sliding_window(self):
        params = self._initialize_sliding_window_params()
        for window in range(params['nwindows']):
            params['leftx_current'], params['rightx_current'] = self._sliding_window_one(window, params)

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(params['left_lane_inds'])
        right_lane_inds = np.concatenate( params['right_lane_inds'])

        return params['nonzero'], left_lane_inds, right_lane_inds

    def _extrapolate_from_last_frame(self):
        img = self.img
        left_fit_polynomial = self.last_frame.left_lane.fit_polynomial
        right_fit_polynomial = self.last_frame.right_lane.fit_polynomial
        margin = self.margin

        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = ((nonzerox > (left_fit_polynomial[0]*(nonzeroy**2) + left_fit_polynomial[1]*nonzeroy +
        left_fit_polynomial[2] - margin)) & (nonzerox < (left_fit_polynomial[0]*(nonzeroy**2) +
        left_fit_polynomial[1]*nonzeroy + left_fit_polynomial[2] + margin)))

        right_lane_inds = ((nonzerox > (right_fit_polynomial[0]*(nonzeroy**2) + right_fit_polynomial[1]*nonzeroy +
        right_fit_polynomial[2] - margin)) & (nonzerox < (right_fit_polynomial[0]*(nonzeroy**2) +
        right_fit_polynomial[1]*nonzeroy + right_fit_polynomial[2] + margin)))

        return nonzero, left_lane_inds, right_lane_inds

    def _fit_polynomial_to_lane_data(self, left_fit_polynomial, right_fit_polynomial):
        img = self.img
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
        left_fitx = left_fit_polynomial[0]*ploty**2 + left_fit_polynomial[1]*ploty + left_fit_polynomial[2]
        right_fitx = right_fit_polynomial[0]*ploty**2 + right_fit_polynomial[1]*ploty + right_fit_polynomial[2]

        left_fitx_mask = (left_fitx >= 0) & (left_fitx <= 1280)
        right_fitx_mask = (right_fitx >= 0) & (right_fitx <= 1280)

        left_fitx = left_fitx[left_fitx_mask].astype(int)
        right_fitx = right_fitx[right_fitx_mask].astype(int)
        left_fity = ploty[left_fitx_mask].astype(int)
        right_fity = ploty[right_fitx_mask].astype(int)

        return left_fitx, right_fitx, left_fity, right_fity

    def find_lane(self):
        img = self.img

        if self.last_frame and self.last_valid_left_lane_found < 12 and self.last_valid_right_lane_found < 12:
            nonzero, left_lane_inds, right_lane_inds = self._extrapolate_from_last_frame()
        else:
            nonzero, left_lane_inds, right_lane_inds = self._sliding_window()

        nonzerox = np.array(nonzero[1])
        nonzeroy = np.array(nonzero[0])

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        if len(leftx) == 0:
            print('no left lane')
        if len(rightx) == 0:
            print('not right lane')

        # Fit a second order polynomial to each
        left_fit_polynomial = np.polyfit(lefty, leftx, 2)
        right_fit_polynomial = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        left_fitx, right_fitx, left_fity, right_fity = self._fit_polynomial_to_lane_data(left_fit_polynomial, right_fit_polynomial)

        left_lane = Lane(leftx, lefty, left_fit_polynomial, left_fitx, left_fity)
        right_lane = Lane(rightx, righty, right_fit_polynomial, right_fitx, right_fity)

        if left_lane.calculate_curvature() < 195:
            # import ipdb; ipdb.set_trace()
            left_lane = self.last_frame.left_lane if self.last_frame else None
            self.last_valid_left_lane_found += 1
        else:
            self.last_valid_left_lane_found = 0

        if right_lane.calculate_curvature() < 195:
            # import ipdb; ipdb.set_trace()
            right_lane = self.last_frame.right_lane if self.last_frame else None
            self.last_valid_right_lane_found += 1
        else:
            self.last_valid_right_lane_found = 0


        return left_lane, right_lane
