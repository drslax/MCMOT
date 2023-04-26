from filterpy.kalman import KalmanFilter
import numpy as np


class Person:
    def __init__(self, idx, box, time, age=1, unmatched_age=0):
        """
        The class Obstacle is defined here. The class has a constructor
            method named __init__ which takes in five arguments
            idx, box, time, age and unmatched_age.
        """
        # set the properties of the obstacle
        self.idx = idx  # obstacle ID
        self.box = box  # bounding box of the obstacle
        self.time = time  # timestamp of when the obstacle was first detected
        self.age = age  # number of times the obstacle was detected
        self.unmatched_age = unmatched_age  # number of times the obstacle was not detected

        # initialize the Kalman filter
        self.kf = KalmanFilter(dim_x=8, dim_z=4)  # create a Kalman filter with 8 states and 4 measurements
        self.kf.x = np.array(
            [self.box[0], 0, self.box[1], 0, self.box[2], 0, self.box[3], 0])  # set initial state of the Kalman filter
        self.kf.P *= 1000  # set the initial state covariance matrix
        Q_std = 0.01  # set the process noise standard deviation
        self.kf.Q[4:, 4:] *= Q_std  # set the process noise covariance matrix

        # set the measurement matrix of the Kalman filter
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0]])
        R_std = 10  # set the measurement noise standard deviation
        self.kf.R[2:, 2:] *= R_std  # set the measurement noise covariance matrix
