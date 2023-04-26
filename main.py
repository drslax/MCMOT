import os

import cv2
import copy
import time
from kalman_filter import Person
from hungarian import associate
from tools import *
from tools_yolo import infer_image
from reid import REID


class LoadVideo:  # for inference
    def __init__(self, path, img_size=(1088, 608)):
        if not os.path.isfile(path):
            raise FileExistsError

        self.cap = cv2.VideoCapture(path)
        self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        print('Length of {}: {:d} frames'.format(path, self.vn))

    def get_VideoLabels(self):
        return self.cap, self.frame_rate, self.vw, self.vh

def main():
    # Initialize global variables
    global idx
    global stored_persons
    # Define minimum duration of obstacle detection in frames
    MIN_DETECTION_DURATION = 3
    # Define maximum gap between observations in frames
    MAX_OBSERVATION_GAP = 10

    images_by_id = dict()
    ids_per_frame = []
    persons_ctr = dict()
    frame_idx = -1

    # Load video
    cap1 = "data/4p-c0.avi"
    cap2 = "data/4p-c1.avi"
    # Load pre-trained YOLO object detection model
    net = cv2.dnn.readNetFromDarknet('yolo/yolov7-tiny.cfg', 'yolo/yolov7-tiny.weights')
    # Initialize variables for video frame dimensions
    height, width = None, None
    # Get current time for calculating time differences
    current_time = time.time()
    all_frames, i = [], 0
    for video in [cap1, cap2]:
        loadvideo = LoadVideo(video)
        video_capture, frame_rate, width, height = loadvideo.get_VideoLabels()
        while True:
            i += 1
            ret, frame = video_capture.read()
            if ret is not True:
                video_capture.release()
                break

            all_frames.append(frame)

    print(len(all_frames))
    # Loop through video frames
    for frame in all_frames:
        frame_idx += 1
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

            # get all images by person
            image_ids = []
            for obj in selected_persons:
                bbox_ = obj.box
                idx_ = obj.idx
                image_ids.append(idx_)
                left, top, right, bottom = convert_data(bbox_)
                color = id_to_color(idx_)
                frame_copy = cv2.rectangle(frame_copy, (int(left), int(top)), (int(right), int(bottom)), color, 2)
                text = "{}".format(idx_)
                frame_copy = cv2.putText(frame_copy, text, (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                if idx_ not in persons_ctr:
                    images_by_id[idx_] = [frame[top:bottom, left:right]]
                    persons_ctr[idx_] = [[frame_idx, left, top, right, bottom]]
                else:
                    images_by_id[idx_].append(frame[top:bottom, left:right])
                    persons_ctr[idx_].append([frame_idx, left, top, right, bottom])
            ids_per_frame.append(set(image_ids))
            stored_persons = copy.deepcopy(new_persons)  # update the stored obstacles with the new obstacle list for future matching

    reid = REID()
    threshold = 360
    exist_ids = set()
    final_fuse_id = dict()

    print(f'Total IDs = {len(images_by_id)}')
    print(f'Total Frames = {len(ids_per_frame)}')
    feats = dict()
    for i in images_by_id:
        print(f'ID number {i} -> Number of frames {len(images_by_id[i])}')
        feats[i] = reid.features_(images_by_id[i])

    for f in ids_per_frame:
        if f:
            if len(exist_ids) == 0:
                for i in f:
                    final_fuse_id[i] = [i]
                exist_ids = exist_ids or f
            else:
                new_ids = f - exist_ids
                for nid in new_ids:
                    dis = []
                    if len(images_by_id[nid]) < 5:
                        exist_ids.add(nid)
                        continue
                    unpickable = []
                    for i in f:
                        for key, item in final_fuse_id.items():
                            if i in item:
                                unpickable += final_fuse_id[key]
                    print('exist_ids {} unpickable {}'.format(exist_ids, unpickable))
                    for oid in (exist_ids - set(unpickable)) & set(final_fuse_id.keys()):
                        tmp = np.mean(reid.compute_distance(feats[nid], feats[oid]))
                        print('nid {}, oid {}, tmp {}'.format(nid, oid, tmp))
                        dis.append([oid, tmp])
                    exist_ids.add(nid)
                    if not dis:
                        final_fuse_id[nid] = [nid]
                        continue
                    if dis[0][1] < threshold:
                        combined_id = dis[0][0]
                        images_by_id[combined_id] += images_by_id[nid]
                        final_fuse_id[combined_id].append(nid)
                    else:
                        final_fuse_id[nid] = [nid]
    print('Final ids and their sub-ids:', final_fuse_id)


    cv2.namedWindow("Original")
    for frame in range(len(all_frames)):
        frame2 = all_frames[frame]
        for idx in final_fuse_id:
            for i in final_fuse_id[idx]:
                for f in persons_ctr[i]:
                    # print('frame {} f0 {}'.format(frame,f[0]))
                    if frame == f[0]:
                        text_scale, text_thickness, line_thickness = get_FrameLabels(frame2)
                        cv2_addBox(idx, frame2, f[1], f[2], f[3], f[4], line_thickness, text_thickness, text_scale)
        cv2.imshow('Original', frame2)
        if cv2.waitKey(50) == ord('q'):
            break
    cv2.destroyAllWindows()


def get_FrameLabels(frame):
    text_scale = max(1, frame.shape[1] / 1600.)
    text_thickness = 1 if text_scale > 1.1 else 1
    line_thickness = max(1, int(frame.shape[1] / 500.))
    return text_scale, text_thickness, line_thickness


def cv2_addBox(track_id, frame, x1, y1, x2, y2, line_thickness, text_thickness, text_scale):
    color = get_color(abs(track_id))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=line_thickness)
    cv2.putText(
        frame, str(track_id), (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=text_thickness)


def write_results(filename, data_type, w_frame_id, w_track_id, w_x1, w_y1, w_x2, w_y2, w_wid, w_hgt):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{x2},{y2},{w},{h}\n'
    else:
        raise ValueError(data_type)
    with open(filename, 'a') as f:
        line = save_format.format(frame=w_frame_id, id=w_track_id, x1=w_x1, y1=w_y1, x2=w_x2, y2=w_y2, w=w_wid, h=w_hgt)
        f.write(line)
    # print('save results to {}'.format(filename))

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

    frame_idx = -1
    print(len(all_frames))
    cv2.namedWindow("Original")
    for frame in all_frames:
        frame_idx += 1
        if len(ids_per_frame) > frame_idx:
            for i in range(len(ids_per_frame[frame_idx])):
                if frame_idx in persons_ctr.keys():
                    for b in persons_ctr[frame_idx]:
                        if b[0] == i:
                            new_idx = b[0]
                            left, top, right, bottom = b[1:]
                            color = id_to_color(new_idx)
                            frame = cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), color, 2)
                            text = "{}".format(new_idx)
                            frame = cv2.putText(frame, text, (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow('Original', frame)
        if cv2.waitKey(50) == ord('q'):
            break
    cv2.destroyAllWindows()


idx = 0
stored_persons = []

if __name__ == "__main__":
    main()
