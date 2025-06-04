from ultralytics import YOLO  # Import the YOLO object detection library
import cv2  # Import OpenCV for video processing
from functions import *  # Import all custom functions from functions.py


model = YOLO("yolo11n.pt")

cap = open_video(r"media/traffic_2.mp4")

setup_window()

"""
Main loop to process each frame of the video
"""
while True:
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:
        break  # Exit loop if there are no more frames

    """
    Process the frame with the model and get annotated frame and object counts
    """
    annotated_frame, counts = process_frame(model, frame)

    """
    Overlay the object counts on the frame
    """
    frame = overlay_counts(annotated_frame, counts)

    """
    Track objects and render their trajectories, directions, and orientations
    """
    annotated_frame, trajectories, directions, orientations = track_and_render_trajectories(
        model(frame), frame
    )

    """
    Display the processed frame in a window
    """
    cv2.imshow("Processed Video", annotated_frame)

    """
    Exit if the 'q' key is pressed
    """
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

"""
Release the video capture object and close all OpenCV windows
"""
cap.release()
cv2.destroyAllWindows()
