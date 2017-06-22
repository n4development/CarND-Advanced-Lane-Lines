
import cv2
import pickle
from binary_image import combined_thresh
from perspective_transform import perspective_transform
from Line import Line


from line_fit import line_fit, tune_fit, final_viz, calc_curve, calc_vehicle_offset
from moviepy.editor import VideoFileClip

class FindLaneLine():
    def __init__(self, window_size, input_video, output_video):
        # init Camera
        with open('calibrate_camera.p', 'rb') as f:
            save_dict = pickle.load(f)
        print("Load calibrate_camera pickle -- done")
        self.input_video = input_video
        self.mtx = save_dict['mtx']
        self.dist = save_dict['dist']
        self.window_size = window_size
        self.left_line = Line(n=window_size)
        self.right_line = Line(n=window_size)
        # init detect lane False
        self.detected = False
        self.left_curve, self.right_curve = 0., 0.  # radius of curvature for left and right lanes
        self.left_lane_curve, self.right_lane_curve = None, None  # for calculating curvature

        # Annotated Input Video
        annotated_v = self.load_video(input_video)
        # generate output Video
        annotated_v.write_videofile(output_video, audio=False)

    def load_video(self, input_video):
        """ Given input_file video, save annotated video to output_file """
        video = VideoFileClip(input_video)
        return video.fl_image(self.annotate_frame)

    def annotate_frame(self, frame):
        #global mtx, dist, left_line, right_line, detected
        #global left_curve, right_curve, left_lane_inds, right_lane_inds

        # Undistort, threshold, perspective transform
        undist = cv2.undistort(frame, self.mtx, self.dist, None, self.mtx)
        img, abs_bin, mag_bin, dir_bin, hls_bin = combined_thresh(undist)
        binary_warped, binary_unwarped, m, m_inv = perspective_transform(img)

        # Perform polynomial fit
        if not self.detected:
            # Slow line fit
            ret = line_fit(binary_warped)
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            nonzerox = ret['nonzerox']
            nonzeroy = ret['nonzeroy']
            left_lane_inds = ret['left_lane_inds']
            right_lane_inds = ret['right_lane_inds']

            # Get moving average of line fit coefficients
            left_fit = self.left_line.add_fit(left_fit)
            right_fit = self.right_line.add_fit(right_fit)

            # Calculate curvature
            self.left_curve, self.right_curve = calc_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy)

            self.detected = True  # slow line fit always detects the line

        else:  # implies detected == True
            # Fast line fit
            left_fit = self.left_line.get_fit()
            right_fit = self.right_line.get_fit()
            ret = tune_fit(binary_warped, left_fit, right_fit)
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            nonzerox = ret['nonzerox']
            nonzeroy = ret['nonzeroy']
            left_lane_inds = ret['left_lane_inds']
            right_lane_inds = ret['right_lane_inds']

            # Only make updates if we detected lines in current frame
            if ret is not None:
                left_fit = ret['left_fit']
                right_fit = ret['right_fit']
                nonzerox = ret['nonzerox']
                nonzeroy = ret['nonzeroy']
                left_lane_inds = ret['left_lane_inds']
                right_lane_inds = ret['right_lane_inds']

                left_fit = self.left_line.add_fit(left_fit)
                right_fit = self.right_line.add_fit(right_fit)
                self.left_curve, self.right_curve = calc_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy)
            else:
                self.detected = False

        vehicle_offset = calc_vehicle_offset(undist, left_fit, right_fit)

        # Perform final visualization on top of original undistorted image
        result = final_viz(undist, left_fit, right_fit, m_inv, self.left_curve, self.right_curve, vehicle_offset)

        return result

if __name__ == '__main__':
    # Annotate the video
    FindLaneLine(9, 'project_video.mp4', 'out.mp4')
