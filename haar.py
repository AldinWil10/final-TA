import cv2
import numpy as np

def haarcascade_frontal_face_detection(cap):
    # Load the Haarcascade classifier
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Capture frames from the webcam
    while True:
        # Capture the current frame
        frame = cap.read()[0]

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the frame
        cv2.imshow("Webcam", frame)

        # Check if the user pressed ESC
        if cv2.waitKey(1) == 27:
            break

if __name__ == "__main__":
    # Create a VideoCapture object
    cap = cv2.VideoCapture(0)

    # Run the Haarcascade face detection algorithm
    haarcascade_frontal_face_detection(cap)

    # Release the VideoCapture object
    cap.release()
