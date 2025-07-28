import os
import cv2
import sys
from deep_sort_realtime.deepsort_tracker import DeepSort
import time

# Parameters (Adjust based on your use-case)
VIDEO_PATH = 'data/input_video.mp4'
DETECTIONS_FILE = 'data/detections.txt'
OUTPUT_FILE = 'results/results_deepsort.txt'
MAX_AGE = 30  # How many frames a track persists without detections

for file in [VIDEO_PATH, DETECTIONS_FILE]:
    if not os.path.exists(file):
        print(f"Error: Required file {file} not found.")
        sys.exit(1)


# Mapping for class IDs if needed.
# YOLOv8 (COCO) 'person' is class_id 0. MOT17 'pedestrian' is class_id 1.
# We will map YOLO's person (0) to MOT17's pedestrian (1) in the output.
CLASS_ID_MAP = {
    0: 1,  # Map YOLO's 'person' (class 0) to MOT17's 'pedestrian' (class 1)
}


# Load detections
def load_detections(detection_file):
    """Load precomputed YOLO detections from file and organize them by frame."""
    detections = {}
    try:
        with open(detection_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 7:
                    frame, x1, y1, x2, y2, conf, cls = line.strip().split(',')
                    frame = int(frame)
                    det = [float(x1), float(y1), float(x2), float(y2), float(conf), int(cls)]
                    detections.setdefault(frame, []).append(det)
    except FileNotFoundError:
        print(f"Error: Detections file not found at {detection_file}. Please ensure detect.py has been run.")
        exit()
    return detections


def main():
    """Run DeepSORT tracker on the input video using precomputed detections and save results in MOT format."""
    tracker = DeepSort(max_age=MAX_AGE)  # tracks persist up to 30 frames
    detections = load_detections(DETECTIONS_FILE)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {VIDEO_PATH}. Please ensure the path is correct.")
        exit()

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    print(f"Starting DeepSORT tracking. Results will be written to {OUTPUT_FILE}")

    with open(OUTPUT_FILE, 'w') as out:
        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_dets = detections.get(frame_id, [])
            input_dets = []
            for det in frame_dets:
                x1, y1, x2, y2, conf, cls = det
                w, h = x2 - x1, y2 - y1

                # Apply class ID mapping for input to DeepSort (if DeepSort uses it)
                # and for consistency in output.
                mapped_cls = CLASS_ID_MAP.get(cls, cls)  # Use mapped ID, or original if not in map

                input_dets.append(([x1, y1, w, h], conf, frame, mapped_cls))

            tracks = tracker.update_tracks(input_dets, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()  # Get bbox in (left, top, right, bottom) format
                x1, y1, x2, y2 = map(int, ltrb)
                w, h = x2 - x1, y2 - y1
                conf = 1.0  # DeepSORT typically doesn't output confidence per track directly, use 1.0 or average detection conf

                # DeepSort's track object doesn't directly expose the class_id from the initial detection.
                # Since we are mapping YOLO's person (0) to MOT17's pedestrian (1) for evaluation,
                # we'll assume all confirmed tracks are of the target class (1).
                output_cls = 1  # Assuming all tracked objects are persons/pedestrians for MOT17 evaluation

                # Write in MOT Challenge format: frame, id, x, y, width, height, confidence, class_id, visibility
                out.write(f"{frame_id},{track_id},{x1},{y1},{w},{h},{conf:.2f},{output_cls},1\n")

            frame_id += 1

    cap.release()
    print(f"DeepSORT tracking complete. Results saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed = end_time - start_time
    tracker_name = "DeepSORT"
    os.makedirs("evaluation_results", exist_ok=True)
    with open("evaluation_results/runtime_log.txt", "a") as f:
        f.write(f"{tracker_name}: {elapsed:.2f}\n")

    print(f"{tracker_name} tracking completed in {elapsed:.2f} seconds.")

