import os
import cv2
import pickle
import random
import argparse
import numpy as np
from fastreid.emb_computer import EmbeddingComputer


def make_parser():
    # Initialization
    parser = argparse.ArgumentParser("Track")

    # Data args
    parser.add_argument("--dataset", type=str, default="mot17")
    parser.add_argument("--pickle_path", type=str, default="../outputs/1. det/MOT17_val.pickle")
    parser.add_argument("--output_path", type=str, default="../outputs/2. det_feat/MOT17_val.pickle")
    parser.add_argument("--data_path", type=str, default="../../dataset/MOT17/train/")
    parser.add_argument("--weight_path", type=str, default="weights/mot17_sbs_S50.pth")

    # Else
    parser.add_argument("--seed", type=float, default=10000)

    return parser


if __name__ == "__main__":
    # Get arguments
    args = make_parser().parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    # Get encoder
    embedder = EmbeddingComputer(dataset=args.dataset, path=args.weight_path)
    # Read detection
    with open(args.pickle_path, 'rb') as f:
        detections = pickle.load(f)

    # Feature extraction
    for vid_name in detections.keys():
        for frame_id in detections[vid_name].keys():
            # If there is no detection
            if detections[vid_name][frame_id] is None:
                continue

            # Read image
            
            img_path = os.path.join(args.data_path, "MOT20-01", 'img1/%06d.jpg' % frame_id)
            # print("img_path: ", img_path)
            img = cv2.imread(img_path)

            # Get detection
            detection = detections[vid_name][frame_id]

            # Get features
            if detection is not None:
                embedding = embedder.compute_embedding(img, detection[:, :4])
                detections[vid_name][frame_id] = np.concatenate([detection, embedding], axis=1)

            # Logging
            print(vid_name, frame_id, flush=True)

    # Save
    with open(args.output_path, 'wb') as handle:
        pickle.dump(detections, handle, protocol=pickle.HIGHEST_PROTOCOL)
