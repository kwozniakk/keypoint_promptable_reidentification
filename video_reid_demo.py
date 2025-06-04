import argparse
import glob
import os

import cv2
import torch

from torchreid.metrics.distance import compute_distance_matrix_using_bp_features
from torchreid.scripts.builder import build_config
from torchreid.tools.feature_extractor import KPRFeatureExtractor


def load_samples(folder):
    image_paths = sorted(
        p for p in glob.glob(os.path.join(folder, '*'))
        if p.lower().endswith(('.jpg', '.jpeg', '.png'))
    )
    if len(image_paths) < 5:
        raise ValueError(f'Need at least 5 reference images in {folder}')

    samples = []
    for path in image_paths[:5]:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f'Failed to read {path}')
        samples.append({'image': img})
    return samples


def process_video(video_path, ref_dir, output_path='output.mp4', config='configs/kpr/imagenet/kpr_occ_posetrack_test.yaml'):
    cfg = build_config(config_path=config)
    cfg.use_gpu = torch.cuda.is_available()
    extractor = KPRFeatureExtractor(cfg)

    ref_samples = load_samples(ref_dir)
    ref_samples, ref_emb, ref_vis, _ = extractor(ref_samples)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f'Cannot open video {video_path}')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        sample = {'image': frame}
        sample, emb, vis, _ = extractor(sample)
        distmat, _ = compute_distance_matrix_using_bp_features(
            emb, ref_emb, vis, ref_vis, use_gpu=cfg.use_gpu)
        dist = distmat.cpu().numpy()[0] / 2
        best_idx = int(dist.argmin())
        cv2.putText(frame, f'Ref {best_idx}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        out.write(frame)

    out.release()
    cap.release()


def main():
    parser = argparse.ArgumentParser(
        description='Run KPR model on a video using reference images')
    parser.add_argument('--video', required=True, help='Input video path')
    parser.add_argument('--ref_dir', required=True, help='Directory with reference images')
    parser.add_argument('--output', default='output.mp4', help='Output video path')
    parser.add_argument(
        '--config',
        default='configs/kpr/imagenet/kpr_occ_posetrack_test.yaml',
        help='Config file for the KPR model')
    args = parser.parse_args()

    process_video(args.video, args.ref_dir, args.output, args.config)


if __name__ == '__main__':
    main()

