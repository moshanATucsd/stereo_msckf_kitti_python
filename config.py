import numpy as np
import cv2

import pykitti

class OptimizationConfigKitti(object):
    """
    Configuration parameters for 3d feature position optimization.
    """
    def __init__(self):
        self.translation_threshold = -1.0  # 0.2
        self.huber_epsilon = 0.01
        self.estimation_precision = 5e-7
        self.initial_damping = 1e-3
        self.outer_loop_max_iteration = 5 # 10
        self.inner_loop_max_iteration = 5 # 10

class ConfigKitti(object):
    def __init__(self):
        basedir = '/media/erl/disk2/kitti'
        date = '2011_09_30'
        drive = '0027'

        # The 'frames' argument is optional - default: None, which loads the whole dataset.
        # Calibration, timestamps, and IMU data are read automatically. 
        # Camera and velodyne data are available via properties that create generators
        # when accessed, or through getter methods that provide random access.
        self.kitti_data = pykitti.raw(basedir, date, drive, frames=range(0, 50, 5))

        # from velo frame to cam frame 
        T_velo_cam0 = self.kitti_data.calib.T_cam0_velo
        T_velo_cam1 = self.kitti_data.calib.T_cam1_velo
        # from velo frame to imu frame 
        T_imu_velo = self.kitti_data.calib.T_velo_imu

        # feature position optimization
        self.optimization_config = OptimizationConfigKitti()

        ## image processor
        self.grid_row = 5
        self.grid_col = 10

        self.grid_num = self.grid_row * self.grid_col

        self.grid_min_feature_num = 5
        self.grid_max_feature_num = 10
        
        self.fast_threshold = 15
        self.ransac_threshold = 3
        self.stereo_threshold = 5
        self.max_iteration = 30
        self.track_precision = 0.01
        self.pyramid_levels = 3
        self.patch_size = 15
        self.win_size = (self.patch_size, self.patch_size)

        self.lk_params = dict(
            winSize=self.win_size,
            maxLevel=self.pyramid_levels,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                self.max_iteration, 
                self.track_precision),
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        

        ## msckf vio
        # gravity
        self.gravity_acc = 9.81
        self.gravity = np.array([0.0, 0.0, -self.gravity_acc])

        # Framte rate of the stereo images. This variable is only used to 
        # determine the timing threshold of each iteration of the filter.
        self.frame_rate = 10

        # Maximum number of camera states to be stored
        self.max_cam_state_size = 20

        # The position uncertainty threshold is used to determine
        # when to reset the system online. Otherwise, the ever-increaseing
        # uncertainty will make the estimation unstable.
        # Note this online reset will be some dead-reckoning.
        # Set this threshold to nonpositive to disable online reset.
        self.position_std_threshold = 8.0

        # Threshold for determine keyframes
        self.rotation_threshold = 0.2618
        self.translation_threshold = 0.4
        self.tracking_rate_threshold = 0.5

        # Noise related parameters (Use variance instead of standard deviation)
        self.gyro_noise = 0.005 ** 2
        self.acc_noise = 0.05 ** 2
        self.gyro_bias_noise = 0.001 ** 2
        self.acc_bias_noise = 0.01 ** 2
        self.observation_noise = 0.035 ** 2

        # initial state
        self.velocity = np.zeros(3)

        # The initial covariance of orientation and position can be
        # set to 0. But for velocity, bias and extrinsic parameters, 
        # there should be nontrivial uncertainty.
        self.velocity_cov = 0.25
        self.gyro_bias_cov = 0.01
        self.acc_bias_cov = 0.01
        self.extrinsic_rotation_cov = 3.0462e-4
        self.extrinsic_translation_cov = 2.5e-5

        ## calibration parameters
        # T_imu_cam: takes a vector from the IMU frame to the cam frame.
        # T_cn_cnm1: takes a vector from the cam0 frame to the cam1 frame.
        # see https://github.com/ethz-asl/kalibr/wiki/yaml-formats
        self.T_imu_cam0 = T_velo_cam0 @ T_imu_velo
        self.cam0_camera_model = 'pinhole'
        self.cam0_distortion_model = 'radtan'
        self.cam0_distortion_coeffs = np.array([[0., 0., 0., 0.]])
        self.cam0_intrinsics = np.array([9.786977e+02, 6.900000e+02, 9.717435e+02, 2.497222e+02])
        self.cam0_resolution = np.array([1392, 512])

        self.T_imu_cam1 = T_velo_cam1 @ T_imu_velo
        self.T_cn_cnm1 = self.T_imu_cam1 @ np.linalg.inv(self.T_imu_cam0)
        self.cam1_camera_model = 'pinhole'
        self.cam1_distortion_model = 'radtan'
        self.cam1_distortion_coeffs = np.array([[0.,  0., 0., 0.]])
        self.cam1_intrinsics = np.array([9.892043e+02, 7.020000e+02, 9.832048e+02, 2.616538e+02])
        self.cam1_resolution = np.array([1392, 512])
        # self.baseline = 

        self.T_imu_body = np.identity(4)

if __name__ == '__main__':

    config = ConfigKitti()
    print("T from imu frame to cam0 frame")
    print(config.T_imu_cam0)
    print("T from imu frame to cam1 frame")
    print(config.T_imu_cam1)
    print("T from cam0 frame to cam1 frame")
    print(config.T_cn_cnm1)