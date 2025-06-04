# Traffic Object Detection and Tracking - Week 1 Report

## Objectives / Tasks

- Use pre-trained YOLO (latest version) for object detection on traffic videos.
- Count the number of objects (cars).
- Implement real-time object tracking:
  - Render car trajectory.
  - Estimate direction and orientation of each car.

## Notes

- Work with two-way traffic videos only (incoming and outgoing cars).
- Use 2D bounding boxes; later map to 3D space.
- Perform optical flow estimation (for trajectory and direction estimation).

## Observations & Learnings

- The system successfully detects and counts multiple object classes (including trucks, cars, and stop signs) in each video frame using the YOLO model.
- The number of detected objects per class is overlaid on the video, providing real-time statistics for each frame.
- The trajectories of moving objects are visualized, showing the paths they follow across frames.
- The direction and orientation (angle) of each tracked object are displayed, indicating their movement trends.
- The video window dynamically updates to reflect these detections and annotations as the video plays.
- The effectiveness of the YOLO model and the tracking logic can be assessed by observing detection accuracy and the continuity of object trajectories.

## Problems / Issues Faced

- Processing each frame with the YOLO model was computationally intensive; tracking via SORT resulted in laggy frames.
- Storing all trajectory points for every object throughout the video can lead to high memory usage for long or heavy videos.
- The tracking method does not handle re-identification when objects leave and re-enter the frame; it treats them as a new object entirely.