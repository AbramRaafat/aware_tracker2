# AdapTrack/trackers/tracker.py

from opts import opt
from trackers.cmc import *
from trackers import metrics
from trackers.units import Track
from trackers import linear_assignment
import numpy as np

class Tracker:
    def __init__(self, metric, vid_name):
        # Set parameters
        self.metric = metric
        
        # Occlusion parameters from opts
        self.max_iou_threshold = opt.max_iou_threshold
        self.proximity_distance = opt.proximity_distance
        self.min_occlusion_frames = opt.min_occlusion_frames
        self.proximity_ratio_threshold = opt.proximity_ratio_threshold
        self.id_switch_penalty = opt.id_switch_penalty

        # Initialization
        self.tracks = []
        self.next_id = 1
        self.cmc = CMC(vid_name)

    @staticmethod
    def compute_iou(bbox1, bbox2):
        """Calculate Intersection over Union between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
        union_area = w1 * h1 + w2 * h2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    @staticmethod
    def get_center(bbox):
        """Convert bounding box to center coordinates"""
        return np.array([bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2])

    def update_occlusion_status(self):
        """Update occlusion state for all tracks"""
        for track in self.tracks:
            if not track.is_confirmed():
                continue

            # Update position history
            current_bbox = track.to_tlwh()
            track.occlusion_history['positions'].append(self.get_center(current_bbox))
            if len(track.occlusion_history['positions']) > self.min_occlusion_frames:
                track.occlusion_history['positions'].pop(0)

            # Initialize occlusion metrics
            max_iou = 0.0
            proximity_ratio = 0.0
            
            # Check against all other tracks
            other_tracks = [t for t in self.tracks if t.track_id != track.track_id]
            if other_tracks:
                # Calculate IOUs
                ious = [self.compute_iou(current_bbox, t.to_tlwh()) for t in other_tracks]
                max_iou = max(ious) if ious else 0.0
                
                # Calculate proximity
                centers = [self.get_center(t.to_tlwh()) for t in other_tracks]
                if centers and track.occlusion_history['positions']:
                    distances = cdist([track.occlusion_history['positions'][-1]], centers)
                    proximity_ratio = np.mean(distances < self.proximity_distance)

            # Update occlusion counter
            if (max_iou > self.max_iou_threshold and 
                len(track.occlusion_history['positions']) >= self.min_occlusion_frames and
                proximity_ratio >= self.proximity_ratio_threshold):
                track.occlusion_history['occlusion_counter'] += 1
            else:
                track.occlusion_history['occlusion_counter'] = 0

            # Set occlusion state
            track.is_occluded = (track.occlusion_history['occlusion_counter'] 
                                >= self.min_occlusion_frames)

    def initiate_track(self, detection):
        new_track = Track(detection.to_cxcyah(), self.next_id, 
                         detection.confidence, detection.feature)
        # Initialize occlusion history
        new_track.occlusion_history = {
            'positions': [self.get_center(new_track.to_tlwh())],
            'occlusion_counter': 0,
            'current_bbox': new_track.to_tlwh()
        }
        new_track.is_occluded = False
        self.tracks.append(new_track)
        self.next_id += 1

    # Rest of the original Tracker methods remain unchanged
    def predict(self):
        for track in self.tracks:
            track.predict()

    def camera_update(self):
        warp_matrix = self.cmc.get_warp_matrix()
        for track in self.tracks:
            apply_cmc(track, warp_matrix)

    def gated_metric(self, tracks, detections, track_indices, detection_indices):
        targets = np.array([tracks[i].track_id for i in track_indices])
        features = np.array([detections[i].feature for i in detection_indices])
        cost_matrix = self.metric.distance(features, targets)
        cost_matrix_min = np.min(cost_matrix)
        cost_matrix_max = np.max(cost_matrix)
        cost_matrix = linear_assignment.gate_cost_matrix(
            cost_matrix, tracks, detections, track_indices, detection_indices
        )
        return cost_matrix, cost_matrix_min, cost_matrix_max

    def match(self, detections):
        self.update_occlusion_status()  # Critical occlusion check
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        matches_a, _, unmatched_detections = linear_assignment.min_cost_matching(
            [self.gated_metric, metrics.iou_constraint, True],
            opt.max_distance, self.tracks, detections, confirmed_tracks
        )

        unmatched_tracks_a = list(set(confirmed_tracks) - set(k for k, _ in matches_a))
        candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1
        ]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1
        ]

        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                [metrics.iou_cost, None, True], 
                opt.max_iou_distance, self.tracks,
                detections, candidates, unmatched_detections
            )

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))

        return matches, unmatched_tracks, unmatched_detections

    def update(self, detections):
        matches, unmatched_tracks, unmatched_detections = self.match(detections)

        # Update matched tracks
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx])

        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # Create new tracks for high-confidence detections
        for detection_idx in unmatched_detections:
            if detections[detection_idx].confidence >= opt.conf_thresh:
                self.initiate_track(detections[detection_idx])

        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update metric features
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
        self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)