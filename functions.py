import cv2
import numpy as np

def open_video(video_path):
    """
    Open and return a video capture object.
    """
    return cv2.VideoCapture(video_path)

def setup_window(window_name="Processed Video", width=1280, height=720):
    """
    Create and resize a named OpenCV window.
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

def process_frame(model, frame):
    """
    Run YOLO model on a frame, return annotated frame and class counts.
    """
    results = model(frame)
    annotated_frame = results[0].plot()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    class_names = model.names
    counts = {}
    for cls_id in class_ids:
        cls_name = class_names.get(cls_id, str(cls_id))
        counts[cls_name] = counts.get(cls_name, 0) + 1
    return annotated_frame, counts

def overlay_counts(frame, counts, y0=30, dy=25):
    """
    Overlay class counts on the frame.
    """
    for i, (cls_name, count) in enumerate(counts.items()):
        text = f"{cls_name}: {count}"
        cv2.putText(frame, text, (10, y0 + i * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame

def track_and_render_trajectories(results, frame, trajectories={}, directions={}, orientations={}, min_track_length=5):
    """
    Detect, track, and render object trajectories using simple centroid tracking.
    Args:
        results: YOLO model results (from process_frame).
        frame: Current video frame (numpy array).
        trajectories: Dict mapping track_id to list of (x, y) positions.
        directions: Dict mapping track_id to direction vector.
        orientations: Dict mapping track_id to orientation angle (degrees).
        min_track_length: Minimum number of points to draw trajectory.
    Returns:
        frame: Annotated frame.
        trajectories, directions, orientations: Updated dicts.
    """
    boxes = results[0].boxes.xyxy.cpu().numpy() if hasattr(results[0].boxes, 'xyxy') else []
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int) if hasattr(results[0].boxes, 'cls') else []
    detections = [(int((box[0]+box[2])/2), int((box[1]+box[3])/2), int(box[0]), int(box[1]), int(box[2]), int(box[3]), class_ids[i] if i < len(class_ids) else -1)
                  for i, box in enumerate(boxes)]

    next_id = max(trajectories.keys(), default=0) + 1
    assigned_tracks = set()
    for cx, cy, *_ in detections:
        min_id, min_dist = None, float('inf')
        for track_id, pts in trajectories.items():
            if pts:
                dist = np.hypot(cx - pts[-1][0], cy - pts[-1][1])
                if dist < min_dist and dist < 50:
                    min_dist, min_id = dist, track_id
        if min_id is not None and min_id not in assigned_tracks:
            trajectories[min_id].append((cx, cy))
            assigned_tracks.add(min_id)
        else:
            trajectories[next_id] = [(cx, cy)]
            next_id += 1

    for track_id, pts in trajectories.items():
        if len(pts) >= min_track_length:
            for j in range(1, len(pts)):
                cv2.line(frame, pts[j - 1], pts[j], (0, 255, 0), 2)
            dx, dy = pts[-1][0] - pts[-min_track_length][0], pts[-1][1] - pts[-min_track_length][1]
            directions[track_id] = (dx, dy)
            angle = np.degrees(np.arctan2(dy, dx))
            orientations[track_id] = angle
            cv2.arrowedLine(frame, pts[-min_track_length], pts[-1], (0, 0, 255), 2, tipLength=0.3)
            cv2.putText(frame, f'{int(angle)} deg', pts[-1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return frame, trajectories, directions, orientations
