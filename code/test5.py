import cv2
import numpy as np
import os
import mediapipe as mp
import pandas as pd

# Load YOLO model
weights_path = "code/yolov3.weights"
config_path = "code/yolov3.cfg"
names_path = "code/coco.names"

net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class names
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# get mp blazepose ready
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils # For drawing keypoints
points = mpPose.PoseLandmark # Landmarks

dir = "pedestrians"
for img_name in os.listdir(dir):
    # Read input image
    path = dir + "/" + img_name
    image = cv2.imread(path)

    # resizing for easier viewing
    scale = 1
    if image.shape[0] > image.shape[1]:
        if image.shape[0] > 650:
            scale = 650 / image.shape[0]
    elif image.shape[1] > image.shape[0]:
        if image.shape[1] > 650:
            scale = 650 / image.shape[1]
    image = cv2.resize(image, None, fx=scale, fy=scale)

    height, width, _ = image.shape

    # Convert image to blob
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Forward pass to get predictions
    outputs = net.forward(output_layers)

    # Process predictions
    boxes, confidences, class_ids = [], [], []
    confidence_threshold = 0.5
    nms_threshold = 0.4

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                center_x, center_y, w, h = (detection[:4] * [width, height, width, height]).astype(int)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    # Draw bounding boxes
    people = []
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        if "person" in label:
            people.append(i)
            color = (0, 255, 0)  # Green box
            # cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            # cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # put blazeposes onto people
    for i in range(len(people)):
        x, y, w, h = boxes[people[i]]
        temp = image[y:y+h, x:x+w]

        # temp = np.zeros((h, w, 3)) # censoring for privacy

        results = pose.process(temp)
        if results.pose_landmarks:
            mpDraw.draw_landmarks(temp, results.pose_landmarks, mpPose.POSE_CONNECTIONS) #draw landmarks on temp
        
        image[y:y+h, x:x+w] = temp

    # Show result
    cv2.imshow(img_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
