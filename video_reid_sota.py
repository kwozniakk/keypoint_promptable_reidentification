import argparse
import glob
import os

import cv2
import numpy as np
import torch

from torchreid.metrics.distance import compute_distance_matrix_using_bp_features
from torchreid.scripts.builder import build_config
from torchreid.tools.feature_extractor import KPRFeatureExtractor

try:
    import openpifpaf
except ImportError as e:
    raise ImportError(
        "openpifpaf is required for automatic keypoint annotation.\n"
        "Install it with: pip install openpifpaf"
    ) from e


def detect_keypoints(img, predictor):
    """Run OpenPifPaf on an RGB image and return keypoints."""
    predictions, _, _ = predictor.numpy_image(img)
    if len(predictions) == 0:
        return np.zeros((17, 3), dtype=float)
    keypoints = predictions[0].data.reshape(-1, 3)
    return keypoints


def load_reference_samples(folder, predictor):
    image_paths = sorted(
        p for p in glob.glob(os.path.join(folder, '*'))
        if p.lower().endswith(('.jpg', '.jpeg', '.png'))
    )
    if len(image_paths) < 5:
        raise ValueError(f'Need at least 5 reference images in {folder}')

    samples = []
    for path in image_paths[:5]:
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise ValueError(f'Failed to read {path}')
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        kps = detect_keypoints(img_rgb, predictor)
        samples.append({'image': img_bgr, 'keypoints_xyc': kps, 'negative_kps': np.zeros((0, 17, 3))})
    return samples


def process_video(video_path, ref_dir, output_path='output.mp4',
                  config='configs/kpr/solider/kpr_occ_duke_test.yaml'):
    cfg = build_config(config_path=config)
    cfg.use_gpu = torch.cuda.is_available()
    extractor = KPRFeatureExtractor(cfg)

    # initialise pifpaf predictor
    pifpaf_predictor = openpifpaf.Predictor()

    ref_samples = load_reference_samples(ref_dir, pifpaf_predictor)
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
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        kps = detect_keypoints(frame_rgb, pifpaf_predictor)
        sample = {'image': frame_bgr, 'keypoints_xyc': kps, 'negative_kps': np.zeros((0, 17, 3))}
        sample, emb, vis, _ = extractor(sample)
        distmat, _ = compute_distance_matrix_using_bp_features(
            emb, ref_emb, vis, ref_vis, use_gpu=cfg.use_gpu)
        dist = distmat.cpu().numpy()[0] / 2
        best_idx = int(dist.argmin())
        cv2.putText(frame_bgr, f'Ref {best_idx}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        out.write(frame_bgr)

    out.release()
    cap.release()


def main():
    parser = argparse.ArgumentParser(
        description='Run KPR model on a video with automatic keypoints')
    parser.add_argument('--video', required=True, help='Input video path')
    parser.add_argument('--ref_dir', required=True, help='Directory with reference images')
    parser.add_argument('--output', default='output.mp4', help='Output video path')
    parser.add_argument(
        '--config',
        default='configs/kpr/solider/kpr_occ_duke_test.yaml',
        help='Config file for the KPR model')
    args = parser.parse_args()

    process_video(args.video, args.ref_dir, args.output, args.config)


if __name__ == '__main__':
    main()
