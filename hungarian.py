from scipy.optimize import linear_sum_assignment
from tools import convert_data
import numpy as np


# Define a function to calculate the Intersection over Union (IoU) between two bounding boxes
def box_iou(box1, box2):
    # Convert box data into a standardized format
    box1 = convert_data(box1)
    box2 = convert_data(box2)
    # Calculate the coordinates of the intersection rectangle
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    # calculate the intersection area of the two boxes
    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)  # calculate the area of box1
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)  # calculate the area of box2
    union_area = (box1_area + box2_area) - inter_area  # calculate the union area of the two boxes
    # calculate the intersection over union (IoU)
    iou = inter_area / float(union_area)
    return iou


# Define a function to calculate the linear cost between two bounding boxes
def c_lin(XA, YA, WA, HA, XB, YB, WB, HB):
    # Define constants
    # (_ , w, h, _) = np.array(dataset_images).shape
    w = 960  # set a constant width for the dataset images
    h = 540  # set a constant height for the dataset images
    Q_dist = np.linalg.norm(np.array([w, h]))  # calculate a normalization factor for the position distance
    Q_shp = w * h  # calculate a normalization factor for the shape distance
    # Calculate the Euclidean distances between the centers and the heights and widths of the two boxes
    d1 = np.linalg.norm(np.array([XA - XB, YA - YB]))  # calculate the position distance
    d2 = np.linalg.norm(np.array([HA - HB, WA - WB]))  # calculate the shape distance
    # if the position distance is zero, set it to a small positive value to avoid division by zero
    if d1 == 0.0:
        d1 = 0.0001
    if d2 == 0.0:
        d2 = 0.0001
    # Calculate and return the cost
    cost = (Q_dist / d1) * (Q_shp / d2)
    return cost

# Define a function to calculate the cost of exponential distance between two bounding boxes
def c_exp(XA, YA, WA, HA, XB, YB, WB, HB):
    # Define constants
    w1 = 0.5  # set weight for the position and size distance
    w2 = 1.5  # set weight for the shape distance
    # Calculate the squared distance ratios of the centers and heights and widths of the two boxes
    p1 = ((XA - XB) / WA) ** 2 + ((YA - YB) / HA) ** 2  # calculate position and size distance
    p2 = abs(HA - HB) / (HA + HB) + abs(WA - WB) / (WA + WB)  # calculate shape distance
    cost = np.exp(-w1 * p1) * np.exp(-w2 * p2)  # calculate the cost using exponential function
    return cost


# Define a function that calculates the cost matrix for each old box to each new box using
#   three different cost metrics: IOU, linear, and exponential.
# The function takes in old_boxes, new_boxes, and optional thresholds for IOU, linear, and exponential costs.
def hungarian_cost(old_boxes, new_boxes, iou_thresh=0.3, linear_thresh=10000, exp_thresh=0.5):
    # Create an empty cost matrix to store the IOU, linear, and exponential costs for each old box to each new box.
    cost_matrix = []
    # Iterate through each old box.
    for box1 in old_boxes:
        # Create an empty row to store the cost of each new box for the current old box.
        row = []
        # Iterate through each new box.
        for box2 in new_boxes:
            # Calculate the IOU cost between the two boxes.
            iou_cost = box_iou(box1, box2)
            # Calculate the center, width, and height of the old box.
            XA = box1[0] + box1[2] * 0.5
            YA = box1[1] + box1[3] * 0.5
            WA = box1[2]
            HA = box1[3]
            # Calculate the center, width, and height of the new box.
            XB = box2[0] + box2[2] * 0.5
            YB = box2[1] + box2[3] * 0.5
            WB = box2[2]
            HB = box2[3]
            # Calculate the linear cost between the two boxes.
            lin_cost = c_lin(XA, YA, WA, HA, XB, YB, WB, HB)
            # Calculate the exponential cost between the two boxes.
            exp_cost = c_exp(XA, YA, WA, HA, XB, YB, WB, HB)
            # If the IOU, linear, and exponential costs all meet their respective thresholds,
            #    append the IOU cost to the row.
            # Otherwise, append 0 to the row.
            if iou_cost >= iou_thresh and lin_cost >= linear_thresh and exp_cost >= exp_thresh:
                row.append(iou_cost)
            else:
                row.append(0)
        # Append the row of costs for the current old box to the cost matrix.
        cost_matrix.append(row)

    # Return the cost matrix.
    return cost_matrix


# Define a function that associates old_boxes with new_boxes using the Hungarian algorithm.
# The function takes in old_boxes and new_boxes.
def associate(old_boxes, new_boxes):
    # Calculate the Hungarian cost matrix.
    iou_matrix = np.array(hungarian_cost(old_boxes, new_boxes))
    # Calculate the optimal assignment of old boxes to new boxes using the Hungarian algorithm.
    hungarian_row, hungarian_col = linear_sum_assignment(-iou_matrix)
    # Create empty lists to store matched old boxes and new boxes, unmatched old boxes, and unmatched new boxes.
    matches = []
    unmatched_trackers = []
    unmatched_detections = []

    # Iterate through each assigned old box and new box.
    for i in range(len(hungarian_row)):
        # get the row index of the i-th match
        x = hungarian_row[i]
        # get the column index of the i-th match
        y = hungarian_col[i]
        # if the i-th match has an IoU score below the threshold,
        #   add the corresponding tracker and detection to the unmatched lists.
        if iou_matrix[x][y] < 0.3:
            unmatched_trackers.append(x)
            unmatched_detections.append(new_boxes[y])
        else:
            # otherwise, add the match to the matches list
            matches.append([x, y])

    for t, track in enumerate(old_boxes):
        # if a track was not assigned to a detection, add it to the unmatched tracks list
        if t not in hungarian_row:
            unmatched_trackers.append(t)

    for d, det in enumerate(new_boxes):
        # if a detection was not assigned to a track, add it to the unmatched detections list
        if d not in hungarian_col:
            unmatched_detections.append(det)

    # return the matched tracks, unmatched tracks, and unmatched detections
    return matches, unmatched_trackers, unmatched_detections
