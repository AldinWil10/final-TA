import cv2
import numpy as np

def yolo_face_detection(cap):
    # Load the YOLO model
    model = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    classes = ["face"]

    # Get the image dimensions
    image_height, image_width, _ = cap.read()[1].shape

    # Capture frames from the webcam
    while True:
        # Capture the current frame
        frame = cap.read()[0]

        # Convert the image to a blob
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)

        # Pass the blob through the YOLO model
        model.setInput(blob)
        detections = model.forward()

        # Extract the bounding boxes from the detections
        boxes = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([image_width, image_height, image_width, image_height])
                boxes.append(box.astype("int"))

        # Draw a rectangle around the faces
        for box in boxes:
            cv2.rectangle(frame, box, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow("Webcam", frame)

        # Check if the user pressed ESC
        if cv2.waitKey(1) == 27:
            break

if __name__ == "__main__":
    # Create a VideoCapture object
    cap = cv2.VideoCapture(0)

    # Run the YOLO face detection algorithm
    yolo_face_detection(cap)

    # Release the VideoCapture object
    cap.release()
