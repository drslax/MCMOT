import cv2
import copy
import time
from kalman_filter import Person
from hungarian import associate
from tools import *
from tools_yolo import infer_image


def main():
    # Initialize global variables
    global idx
    global stored_persons
    # Define minimum duration of obstacle detection in frames
    MIN_DETECTION_DURATION = 3
    # Define maximum gap between observations in frames
    MAX_OBSERVATION_GAP = 70

    # Load video
    cap = cv2.VideoCapture("data/crowdUp.mp4")
    # Load pre-trained YOLO object detection model
    net = cv2.dnn.readNetFromDarknet('yolo/yolov7-tiny.cfg', 'yolo/yolov7-tiny.weights')
    # Initialize variables for video frame dimensions
    height, width = None, None
    # Get current time for calculating time differences
    current_time = time.time()
    # Loop through video frames
    while True:
        # Read video frame
        ret, frame = cap.read()
        # If video has ended, break loop
        if not ret:
            break
        # Get video frame dimensions if not already obtained
        if width is None or height is None:
            height, width = frame.shape[:2]
        # Create a copy of the frame to use for drawing bounding boxes
        frame_copy = copy.copy(frame)
        # Use YOLO to detect objects in the frame
        bboxes, _ = infer_image(net, height, width, frame)
        # If this is the first frame, initialize stored_obstacles with detected objects
        if idx == 0:
            stored_persons = []
            for box in bboxes:
                # Create obstacle object with initial values
                obj = Person(idx, box, current_time)
                # Add object to list of stored obstacles
                stored_persons.append(obj)
                # Increment global index variable
                idx += 1
        # For subsequent frames, associate detected objects with previous objects and update their states
        else:
            # Get list of previous object boxes
            previous_boxes = [obj.box for obj in stored_persons]
            # Associate detected objects with previous objects by using Hungarian Algorithm
            matches, unmatched_trackers, unmatched_detections = associate(previous_boxes, bboxes)
            # Initialize lists for new and selected obstacles
            new_persons = []
            selected_persons = []

            # Update matched objects
            for match in matches:
                # Get obstacle object and increment age and reset unmatched age
                obj = stored_persons[match[0]]
                obj.age += 1
                obj.unmatched_age = 0
                # Update object's state with Kalman Filter
                measurement = np.array(bboxes[match[1]])
                obj.kf.update(measurement)
                # Prediction
                dt = current_time - obj.time
                obj.kf.F = return_F_with_dt(dt)
                obj.kf.predict()
                # Update object's time and box for future match
                obj.time = current_time
                obj.box = get_box_from_state(obj.kf.x)
                # Append object to new obstacles list
                new_persons.append(obj)
                # If object has been detected for at least MIN_DETECTION_DURATION frames, add to selected obstacles list
                if obj.age >= MIN_DETECTION_DURATION:
                    selected_persons.append(obj)

            # Add unmatched detected objects as new obstacles with new IDs
            for new_box in unmatched_detections:
                obj = Person(idx, new_box, current_time)
                new_persons.append(obj)
                idx += 1

            # Update unmatched trackers
            for index in unmatched_trackers:
                obj = stored_persons[index]
                obj.unmatched_age += 1
                # prediction step of the Kalman filter using the time since the last observation
                dt = current_time - obj.time
                obj.kf.F = return_F_with_dt(dt)  # update the state transition matrix
                obj.kf.predict()  # predict the new state
                # Update object's time and box for future match
                obj.time = current_time  # update the last observed time of the obstacle
                obj.box = get_box_from_state(obj.kf.x)  # update the obstacle's bounding box using the Kalman filter's state estimation
                if obj.unmatched_age < MAX_OBSERVATION_GAP:  # if the obstacle has been unmatched for too long, discard it
                    selected_persons.append(obj)  # add the obstacle to the selected list for drawing
                    new_persons.append(obj)  # add the obstacle to the new obstacle list for future matching

            # draw bounding boxes on image
            for obj in selected_persons:
                new_idx = obj.idx
                box = obj.box
                left, top, right, bottom = convert_data(box)
                color = id_to_color(new_idx)
                frame_copy = cv2.rectangle(frame_copy, (int(left), int(top)), (int(right), int(bottom)), color, 2)
                text = "{}".format(new_idx)
                frame_copy = cv2.putText(frame_copy, text, (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            stored_persons = copy.deepcopy(new_persons)  # update the stored obstacles with the new obstacle list for future matching
        cv2.imshow('Original', frame_copy)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()


idx = 0
stored_persons = []

if __name__ == "__main__":
    main()
