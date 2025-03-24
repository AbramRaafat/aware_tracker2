#!/usr/bin/env python3

import os
import argparse
import numpy as np
import json
import cv2
from pathlib import Path
import shutil

def parse_args():
    """Parse command line arguments for the MOT20 to COCO converter."""
    parser = argparse.ArgumentParser(description='Convert MOT20 dataset to COCO format')
    parser.add_argument('--mot20_root', type=str, required=True,
                        help='Path to MOT20 dataset root directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory where to save COCO format output')
    return parser.parse_args()

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)

def main():
    args = parse_args()
    
    # Define paths
    mot20_path = args.mot20_root
    output_dir = args.output_dir
    ann_dir = os.path.join(output_dir, 'annotations')
    img_train_dir = os.path.join(output_dir, 'images', 'train')
    img_val_dir = os.path.join(output_dir, 'images', 'val')
    
    # Create necessary directories
    ensure_dir(ann_dir)
    ensure_dir(img_train_dir)
    ensure_dir(img_val_dir)
    
    # Initialize COCO format dictionaries for each split
    splits = ['train', 'val', 'test']
    coco_data = {
        split: {
            'images': [], 
            'annotations': [], 
            'videos': [],
            'categories': [{'id': 1, 'name': 'pedestrian'}]
        } for split in splits
    }
    
    # Counters for each split
    counters = {
        split: {
            'image_cnt': 0,
            'ann_cnt': 0,
            'video_cnt': 0
        } for split in splits
    }
    
    # Process both train and test directories from MOT20
    for data_folder in ['train', 'test']:
        data_path = os.path.join(mot20_path, data_folder)
        print(f"Processing {data_folder} data from {data_path}")
        # Skip if folder doesn't exist
        if not os.path.exists(data_path):
            print(f"Warning: {data_path} does not exist. Skipping.")
            continue
            
        seqs = os.listdir(data_path)
        print(f"Found {len(seqs)} sequences in {data_folder}")
        for seq in sorted(seqs):
            # Skip hidden files
            if seq.startswith('.'):
                continue
                
            seq_path = os.path.join(data_path, seq)
            img_path = os.path.join(seq_path, 'img1')
            det_path = os.path.join(seq_path, 'gt', 'gt.txt')
            
            # Skip if not a directory
            if not os.path.isdir(seq_path):
                continue
                
            print(f"Processing {seq} in {data_folder}")
            if data_folder == 'test':
                current_splits = ['val', 'test']
            else:
                current_splits = ['train']
                
            # Process this sequence for each of its target splits
            for split in current_splits:
                # Increment video counter for this split
                counters[split]['video_cnt'] += 1
                video_id = counters[split]['video_cnt']
                
                # Add video info
                coco_data[split]['videos'].append({
                    'id': video_id, 
                    'file_name': seq
                })
                
                # Get all images in the sequence
                images = [img for img in os.listdir(img_path) if img.endswith('.jpg')]
                num_images = len(images)
                
                print(f"Processing {seq} for {split} split: {num_images} images")
                
                # Load detection data
                try:
                    dets = np.loadtxt(det_path, dtype=np.float32, delimiter=',')
                    print(f"  Loaded {len(dets)} detections from {det_path}")
                except Exception as e:
                    print(f"  Error loading detections from {det_path}: {e}")
                    dets = np.array([])
                
                # Create image entries and copy images
                for i in range(num_images):
                    img_file = f"{i+1:06d}.jpg"
                    img_path_full = os.path.join(img_path, img_file)
                    
                    # Read image to get dimensions
                    img = cv2.imread(img_path_full)
                    if img is None:
                        print(f"  Warning: Could not read {img_path_full}. Skipping.")
                        continue
                        
                    height, width = img.shape[:2]
                    
                    # Image ID in the COCO dataset
                    image_id = counters[split]['image_cnt'] + 1
                    counters[split]['image_cnt'] += 1
                    
                    # Add image info to COCO data
                    image_info = {
                        'file_name': f"{seq}/img1/{img_file}",
                        'id': image_id,
                        'frame_id': i + 1,  # Frame ID in the sequence, starting from 1
                        'prev_image_id': image_id - 1 if i > 0 else -1,
                        'next_image_id': image_id + 1 if i < num_images - 1 else -1,
                        'video_id': video_id,
                        'height': height,
                        'width': width
                    }
                    coco_data[split]['images'].append(image_info)
                    
                    # Copy image to the output directory
                    if split == 'train':
                        dst_dir = os.path.join(img_train_dir, seq, 'img1')
                    else:  # val or test
                        dst_dir = os.path.join(img_val_dir, seq, 'img1')
                    
                    ensure_dir(dst_dir)
                    dst_path = os.path.join(dst_dir, img_file)
                    
                    # Only copy if the destination doesn't exist
                    if not os.path.exists(dst_path):
                        shutil.copy(img_path_full, dst_path)
                
                # Process detections for this sequence
                if len(dets) > 0:
                    # Filter detections for this sequence's frames
                    frame_ids = set(int(det[0]) for det in dets)
                    print(f"  Found detections for {len(frame_ids)} frames")
                    
                    # Add annotations from detections
                    for det in dets:
                        frame_id = int(det[0])
                        
                        # Find the corresponding image_id
                        target_img = None
                        for img in coco_data[split]['images']:
                            if img['video_id'] == video_id and img['frame_id'] == frame_id:
                                target_img = img
                                break
                        
                        if target_img is None:
                            # This can happen if we couldn't read the image earlier
                            continue
                        
                        
                        if not (int(det[6]) == 1):  # whether ignore.
                            continue
                        if int(det[7]) in [3, 4, 5, 6, 9, 10, 11]:  # Non-person
                            continue
                        # if int(det[7]) in [2, 7, 8, 12]:  # Ignored person
                        #     # category_id = -1
                        #     continue
                        # else:
                        #     category_id = 1  # pedestrian(non-static)
                        # Extract bounding box (x, y, width, height)
                        bbox = det[2:6].tolist()
                        
                        # Verify bbox is within image dimensions
                        # if bbox[0] < 0 or bbox[1] < 0 or bbox[0] + bbox[2] > target_img['width'] or bbox[1] + bbox[3] > target_img['height']:
                        #     print(f"  Warning: Invalid bbox {bbox} for frame {frame_id}. Adjusting...")
                        #     # Adjust bounding box to be within image
                        #     bbox[0] = max(0, bbox[0])
                        #     bbox[1] = max(0, bbox[1])
                        #     bbox[2] = min(bbox[2], target_img['width'] - bbox[0])
                        #     bbox[3] = min(bbox[3], target_img['height'] - bbox[1])
                        
                        # Calculate area
                        area = float(bbox[2] * bbox[3])
                        
                        # Skip if bbox is too small
                        if area <= 0:
                            continue
                        
                        # Create annotation
                        ann_id = counters[split]['ann_cnt'] + 1
                        counters[split]['ann_cnt'] += 1
                        
                        ann = {
                            'id': ann_id,
                            'category_id': 1,  # Pedestrian
                            'image_id': target_img['id'],
                            'track_id': 0,
                            'bbox': bbox,
                            'conf': float(det[6]),  # Preserve confidence score
                            'iscrowd': 0,
                            'area': area
                        }
                        
                        coco_data[split]['annotations'].append(ann)
    
    # Write COCO JSONs
    for split in splits:
        out_path = os.path.join(ann_dir, f'{split}.json')
        print(f"Writing {split}.json with {len(coco_data[split]['images'])} images and {len(coco_data[split]['annotations'])} annotations")
        with open(out_path, 'w') as f:
            json.dump(coco_data[split], f)

if __name__ == '__main__':
    main()