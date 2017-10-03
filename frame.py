import numpy as np
import cv2
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class Frame(object):
    def __init__(self, img):
        # self.fname = fname
        self.rgb = img

    # undistort

    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def set_undistort_params(self, undistort_params):
        self.mtx = undistort_params[0]
        self.dist = undistort_params[1]

    def set_perspective_params(self, perspective_params):
        self.M = self._calculate_perspective_transform_matrix(perspective_params)
        self.Minv = self._calculate_perspective_transform_matrix(perspective_params[::-1])

    # conver to different color formats

    def gray(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def hls(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    def hsv(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # color thresholding

    def hls_color_thresh(self, channel, thresh):
        return self._mask(self.hls[:,:,channel], thresh)

    # shared mask function for both color and gradient thresholding

    def _mask(self, img, thresh):
        binary = np.zeros_like(img)
        binary[(img >= thresh[0]) & (img <= thresh[1])] = 1
        return binary

    # sobel thresholding

    def abs_sobel_thresh(self, orient='x', sobel_kernel=3, thresh=(0, 255)):
        sobel = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel) if orient == 'x' else cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        sobel_abs = np.absolute(sobel)

        sobel_scaled = np.uint8(sobel_abs / np.max(sobel_abs) * 255.)

        mask = self._mask(sobel_scaled, thresh)

        return mask

    def mag_thresh(self, sobel_kernel=3, thresh=(0, 255)):
        sobelx = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        sobel_mag = np.sqrt(sobelx**2 + sobely**2)

        sobel_scaled = np.uint8(255. * sobel_mag / np.max(sobel_mag))

        # import ipdb; ipdb.set_trace()

        mask = self._mask(sobel_scaled, thresh)
        return mask

    def dir_threshold(self, sobel_kernel=3, thresh=(0, np.pi/2)):
        sobelx = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        sobel_dir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

        mask = self._mask(sobel_dir, thresh)
        return mask


    def region_of_interest(self, img):
        Y = img.shape[0]
        X = img.shape[1]

        # region of interest
        BOTTOM_LEFT = (int(150 / 960. * X),Y)
        TOP_LEFT = (int(450 / 960. * X), int(310 / 540. * Y))
        TOP_RIGHT = (int(500 / 960. * X), int(310 / 540. * Y))
        BOTTOM_RIGHT = (int(920 / 960. * X),Y)
        vertices = np.array([[BOTTOM_LEFT, TOP_LEFT, TOP_RIGHT, BOTTOM_RIGHT]], dtype=np.int32)

        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon formed from
        `vertice`. The rest of the image is set to black
        """

        # return an array of zeros with the same shape and type as a given array
        mask = np.zeros_like(img)

        if len(img.shape) > 2:
            channel_count = img.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        #filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    # Perspective transform

    def _calculate_perspective_transform_matrix(self, perspective_params):
        src_pts = perspective_params[0]
        dst_pts = perspective_params[1]

        return cv2.getPerspectiveTransform(src_pts, dst_pts)

    def perspective_transform(self, img):
        size = img.shape[1], img.shape[0]
        return cv2.warpPerspective(img, self.M, size)

    # draw lane line
    def draw_lane_line(self, lane_tracker):
        # import ipdb; ipdb.set_trace()
        img = self.perspective_transformed
        left_lane, right_lane = lane_tracker.find_lane()

        self.left_lane = left_lane
        self.right_lane = right_lane

        out_img = np.dstack((img, img, img))*255

        out_img[left_lane.fity,left_lane.fitx] = [255, 255, 0]
        out_img[right_lane.fity,right_lane.fitx] = [255, 255, 0]

        return out_img

    # unwarping

    def draw_unwarped_lane(self):
        left_fitx = self.left_lane.fitx
        left_fity = self.left_lane.fity
        right_fitx = self.right_lane.fitx
        right_fity = self.right_lane.fity

        original = self.undistorted
        warped = self.perspective_transformed

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, left_fity]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, right_fity])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (warped.shape[1], warped.shape[0]))
        # Combine the result with the original image
        # import ipdb; ipdb.set_trace()
        result = cv2.addWeighted(original, 1, newwarp, 0.3, 0)

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        left_lane_curvature = (800,100)
        right_lane_curvature = (800,150)
        offset_center = (800, 200)
        fontScale              = 1
        fontColor              = (0,0,0)
        lineType               = 2
        pt1 = (780, 50)
        pt2 = (1280, 230)

        cv2.rectangle(result, pt1, pt2, (255, 255, 255), thickness = -1)

        cv2.putText(result,"left lane curvature: " + str(self.left_lane.calculate_curvature()),
            left_lane_curvature,
            font,
            fontScale,
            fontColor,
            lineType)

        cv2.putText(result,"right lane curvature: " + str(self.right_lane.calculate_curvature()),
            right_lane_curvature,
            font,
            fontScale,
            fontColor,
            lineType)

        # plt.imshow(result)

        # import ipdb; ipdb.set_trace()

        x_closest_left = self.left_lane.closest_to_car()
        x_closest_right = self.right_lane.closest_to_car()

        center = (x_closest_left + x_closest_right) // 2

        offset = (center - 640) * 3.7/570

        cv2.putText(result,"offset: " + "{0:.2f}".format(offset),
            offset_center,
            font,
            fontScale,
            fontColor,
            lineType)

        return result

    # write images

    def write_images(self):
        write_original = True
        write_color_conversion = True
        write_masks = True
        write_perspective_transform = True
        write_lane = True

        color_conversion_map = {

            '01_undistorted': self.undistorted,
            '01a_rgb_r': self.undistorted[:,:,0],
            '01b_rgb_g': self.undistorted[:,:,1],
            '01c_rgb_b': self.undistorted[:,:,2],
            '02_gray': self.gray,
            '03_hls': self.hsv,
            '03a_hsv_h': self.hsv[:,:,0],
            '03b_hsv_s': self.hsv[:,:,1],
            '03c_hsv_v': self.hsv[:,:,2],
            '04_hls': self.hls,
            '04a_hls_h': self.hls[:,:,0],
            '04b_hls_l': self.hls[:,:,1],
            '04c_hls_s': self.hls[:,:,2]
        }

        overwrite_map = {
            '01a_rgb_r': False,
            '01b_rgb_g': False,
            '01c_rgb_b': False,
            '01_undistorted': False,
            '02_gray': False,
            '03_hls': False,
            '03a_hsv_h': False,
            '03b_hsv_s': False,
            '03c_hsv_v': False,
            '04_hls': False,
            '04a_hls_h': False,
            '04b_hls_l': False,
            '04c_hls_s': False,
            '05_hls_color_mask': True,
            '06_sobel_mask': True,
            '07a_color_binary': True,
            '07b_combined_binary': True,
            '07c_region_of_interest': True,
            '08_perspective_transformed': True,
            '09_lane_sliding_window': True,
            '10_unwarped_lane': True
        }

        # original
        if write_original:
            self._write(self.rgb, '00_rgb', overwrite = False)

        # color conversion
        if write_color_conversion:
            for key, value in color_conversion_map.items():
                self._write(value, key, overwrite = overwrite_map[key])

        # thresholding
        if write_masks:
            self._write(self.color_masks,'05_hls_color_mask', overwrite = overwrite_map['05_hls_color_mask'])

            self._write(self.sobel_masks,'06_sobel_mask', overwrite = overwrite_map['06_sobel_mask'])

            self._write(self.color_binary,'07a_color_binary', overwrite = overwrite_map['07a_color_binary'])

            self._write(self.combined_binary,'07b_combined_binary', overwrite = overwrite_map['07b_combined_binary'])

            self._write(self.region_of_interest,'07c_region_of_interest', overwrite = overwrite_map['07c_region_of_interest'])

        # Perspective transform
        if write_perspective_transform:
            self._write(self.perspective_transformed,'08_perspective_transformed', overwrite = overwrite_map['08_perspective_transformed'])

        if write_lane:
            self._write(self.lane,'09_lane_sliding_window', overwrite = overwrite_map['09_lane_sliding_window'])
            self._write(self.unwarped_lane,'10_unwarped_lane', overwrite = overwrite_map['10_unwarped_lane'])

    def _write(self, img, fname, overwrite = False, gray=False):
        from datetime import datetime
        import calendar

        d = datetime.utcnow()
        # unixtime = calendar.timegm(d.utctimetuple())
        unixtime = 0

        output_dir = 'output_images/'
        output_name = str(unixtime) + '_' + fname + ".jpg"

        if overwrite or output_name not in os.listdir(output_dir):
            if len(img.shape) == 2:
                mpimg.imsave(output_dir + output_name, img, cmap='gray')
            else:
                mpimg.imsave(output_dir + output_name, img)
