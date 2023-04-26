import cv2
import numpy as np


def generate_boxes(outs, height, width, tconf):
    # Initialize empty lists for storing boxes, confidences, class ids, and centers
    boxes = []
    confidences = []
    class_ids = []
    centers = []

    # Loop through each output in outs
    for out in outs:
        # Loop through each detection in the output
        for detection in out:
            # Extract class scores and class ID from the detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # Check if the confidence of the detection is greater than the threshold and the class ID is 0 (person)
            if confidence > tconf and class_id == 0:
                # Extract the bounding box coordinates
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, bwidth, bheight = box.astype('int')
                # Calculate the top left corner of the bounding box
                x = int(centerX - (bwidth / 2))
                y = int(centerY - (bheight / 2))
                # Append the box coordinates, confidence, and class ID to the corresponding lists
                boxes.append([x, y, int(bwidth), int(bheight)])
                centers.append(np.array([[x], [y]]))
                confidences.append(float(confidence))
                class_ids.append(class_id)
    # Apply non-maximum suppression to remove redundant bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    boxes_ = []
    confidences_ = []
    for i in range(len(boxes)):
        if i in indices:
            boxes_.append(boxes[i])
            confidences_.append(confidences[i])

    # Return the filtered bounding boxes and their corresponding confidences
    return boxes_, confidences_


def infer_image(net, height, width, img):
    # Get the names of the output layers in the network
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    # Convert the input image to a blob
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    # Set the input of the network to the blob and forward propagate it through the network
    net.setInput(blob)
    outs = net.forward(layer_names)
    # Generate bounding boxes from the network output
    bboxes = generate_boxes(outs, height, width, 0.5)
    # Return the bounding boxes
    return bboxes