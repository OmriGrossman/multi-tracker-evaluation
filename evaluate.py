"""
evaluate.py - Compare tracker outputs against ground truth using MOT metrics.
Also logs runtime and FPS per tracker.
"""

import os
import motmetrics as mm
import numpy as np
import pandas as pd

GT_FILE = 'data/gt.txt'
RUNTIME_LOG = 'evaluation_results/runtime_log.txt'
OUTPUT_CSV = 'evaluation_results/tracker_comparison.csv'

TRACKERS = {
    'DeepSORT': 'results/results_deepsort.txt',
    'BOTSort': 'results/results_botsort.txt',
    'ByteTrack': 'results/results_bytetrack.txt'
}

os.makedirs('evaluation_results', exist_ok=True)


def load_tracking_results(file_path):
    """Load tracking results or ground truth from a CSV file into a DataFrame and compute derived coordinates."""
    data = pd.read_csv(file_path, header=None,
                       names=['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis'])
    data['x2'] = data['x'] + data['w']
    data['y2'] = data['y'] + data['h']
    return data


def calculate_mot_metrics(gt, predictions):
    """Compute MOT metrics for a single tracker given the ground truth and predictions."""
    acc = mm.MOTAccumulator(auto_id=True)
    for frame_id in sorted(gt['frame'].unique()):
        gt_boxes = gt[gt['frame'] == frame_id][['x', 'y', 'x2', 'y2']].values
        gt_ids = gt[gt['frame'] == frame_id]['id'].values

        pred_boxes = predictions[predictions['frame'] == frame_id][['x', 'y', 'x2', 'y2']].values
        pred_ids = predictions[predictions['frame'] == frame_id]['id'].values

        distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        acc.update(gt_ids, pred_ids, distances)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=['mota', 'motp', 'num_switches', 'num_fragmentations', 'mostly_tracked', 'mostly_lost'],
        name='acc'
    )
    return summary


def read_runtimes(log_path):
    """Parse the runtime log file and return a dictionary of runtimes by tracker name."""
    runtimes = {}
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            for line in f:
                try:
                    name, time_str = line.strip().split(':')
                    runtimes[name.strip().lower()] = float(time_str.strip())
                except ValueError:
                    continue
    return runtimes



def main():
    """Main function that loads GT, evaluates each tracker, and printssaves the metric summary."""
    print("Loading ground truth...")
    gt = load_tracking_results(GT_FILE)
    runtimes = read_runtimes(RUNTIME_LOG)

    all_metrics = []

    for tracker_name, tracker_file in TRACKERS.items():
        print(f"Processing tracker: {tracker_name}")
        predictions = load_tracking_results(tracker_file)
        mot_summary = calculate_mot_metrics(gt, predictions)

        frame_count = predictions['frame'].max() + 1
        runtime_sec = runtimes.get(tracker_name.lower(), None)

        tracker_metrics = {
            'Tracker': tracker_name,
            'MOTA': mot_summary['mota']['acc'],
            'MOTP': mot_summary['motp']['acc'],
            'ID Switches': mot_summary['num_switches']['acc'],
            'Fragmentations': mot_summary['num_fragmentations']['acc'],
            'Mostly Tracked (%)': mot_summary['mostly_tracked']['acc'],
            'Mostly Lost (%)': mot_summary['mostly_lost']['acc'],
            'Average Track Length': predictions.groupby('id').size().mean(),
            'Estimated ID Switches': predictions.groupby('id')['frame']
                .apply(lambda x: np.sum(np.diff(sorted(x)) > 1)).sum(),
            'Runtime (s)': runtime_sec if runtime_sec else 'NA',
        }

        all_metrics.append(tracker_metrics)

    comparison_df = pd.DataFrame(all_metrics)
    comparison_df.to_csv(OUTPUT_CSV, index=False)

    print("\nComparative Tracker Evaluation Metrics:")
    print(comparison_df.to_string(index=False))


if __name__ == '__main__':
    main()
