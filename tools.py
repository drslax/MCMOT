import numpy as np


# This function takes an ID as input and returns
# a unique RGB color tuple that is derived from the ID.
def id_to_color(idx):
    blue = idx * 5 % 256
    green = idx * 36 % 256
    red = idx * 23 % 256
    return red, green, blue


# This function takes a bounding box array as input and returns four integers
# representing the top-left and bottom-right corners of the bounding box.
def convert_data(box):
    x1 = int(box[0])
    x2 = int(box[0] + box[2])
    y1 = int(box[1])
    y2 = int(box[1] + box[3])
    return x1, y1, x2, y2


# This function returns a state transition matrix that includes the time delta (dt) between image frames.
# The value of dt is hard-coded to 1/25, which is the FPS for the video used in this application (60FPS).
# The state transition matrix is an 8x8 identity matrix with non-zero values only in certain locations.
def return_F_with_dt(dt):
    F = np.eye(8)
    F[0, 1] = dt
    F[2, 3] = dt
    F[4, 5] = dt
    F[6, 7] = dt
    return F


# This function takes a state vector as input and returns a bounding box array.
# The state vector is an 8x1 array representing the position and velocity of an object.
# The bounding box array contains four values representing the top-left corner coordinates and width/height of the object.
def get_box_from_state(state):
    return [state[0], state[2], state[4], state[6]]
