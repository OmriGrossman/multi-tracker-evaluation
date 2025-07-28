import os
import sys
from ultralytics import YOLO
import time

# Hyperparameters (Adjust based on your use-case)
VIDEO_PATH = 'data/input_video.mp4'
OUTPUT_FILE = 'results/results_botsort.txt'
YOLO_MODEL = 'yolov8s.pt'  # YOLO model variant
CONFIDENCE_THRESHOLD = 0.3  # Detection confidence threshold
TRACKER_CONFIG = 'botsort.yaml'

if not os.path.exists(VIDEO_PATH):
    print(f"Error: Video file {VIDEO_PATH} not found.")
    sys.exit(1)


# Mapping for class IDs if needed.
# YOLOv8 (COCO) 'person' is class_id 0. MOT17 'pedestrian' is class_id 1.
# We will map YOLO's person (0) to MOT17's pedestrian (1) in the output.
CLASS_ID_MAP = {
    0: 1,  # Map YOLO's 'person' (class 0) to MOT17's 'pedestrian' (class 1)
}


def run_ultralytics_botsort():
    """Use Ultralytics built-in BOTSort"""
    print("Running BOTSort via Ultralytics...")

    model = YOLO('yolov8n.pt')  # You can change this to yolov8s.pt, etc.

    # Run tracking with Ultralytics and BOTSort
    results = model.track(
        source=VIDEO_PATH,
        tracker=TRACKER_CONFIG,
        conf=CONFIDENCE_THRESHOLD,
        save=False,
        stream=True,
        verbose=False
    )

    all_tracks = []

    for frame_id, result in enumerate(results):
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xywh.cpu().numpy()  # [x_center, y_center, width, height]
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int) # Get original class IDs from YOLO

            for i in range(len(boxes)):
                x_center, y_center, w, h = boxes[i]
                x1 = x_center - w / 2
                y1 = y_center - h / 2

                original_cls = classes[i]
                # Apply class ID mapping for output
                mapped_cls = CLASS_ID_MAP.get(original_cls, original_cls) # Use mapped ID, or original if not in map


                all_tracks.append({
                    'frame_id': frame_id,
                    'track_id': track_ids[i],
                    'bbox': [x1, y1, w, h], # Store as x,y,w,h
                    'confidence': confidences[i],
                    'class_id': mapped_cls # Store the mapped class ID
                })

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        for track in all_tracks:
            # Write in MOT Challenge format: frame, id, x, y, width, height, confidence, class_id, visibility
            f.write(f"{track['frame_id']},{track['track_id']},{track['bbox'][0]:.2f},{track['bbox'][1]:.2f},"
                    f"{track['bbox'][2]:.2f},{track['bbox'][3]:.2f},{track['confidence']:.3f},{track['class_id']},1\n")

    print(f"BOTSort tracking complete. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    start_time = time.time()
    run_ultralytics_botsort()
    end_time = time.time()
    elapsed = end_time - start_time
    tracker_name = "BOTSORT"
    os.makedirs("evaluation_results", exist_ok=True)
    with open("evaluation_results/runtime_log.txt", "a") as f:
        f.write(f"{tracker_name}: {elapsed:.2f}\n")

    print(f"{tracker_name} tracking completed in {elapsed:.2f} seconds.")


