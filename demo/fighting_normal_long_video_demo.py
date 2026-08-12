# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import mmengine
import numpy as np
import torch
from mmengine import Config, DictAction
from mmengine.dataset import Compose

from mmaction.apis import inference_recognizer, init_recognizer

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.9
FONT_THICKNESS = 2
TEXT_ORIGIN = (16, 36)
BOX_PADDING = 8

EXCLUDED_PIPELINE_STEPS = {
    'OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode', 'PyAVInit',
    'PyAVDecode', 'RawFrameDecode', 'SampleFrames'
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run fighting/normal recognition on a long video and '
        'write the current label on the output video.')
    parser.add_argument('config', help='model config file')
    parser.add_argument('checkpoint', help='trained checkpoint file')
    parser.add_argument('video', help='input long video file')
    parser.add_argument('label', help='label map file, one class per line')
    parser.add_argument('out_file', help='output video file, e.g. result.mp4')
    parser.add_argument(
        '--device', default='cuda:0', help='device, e.g. cuda:0 or cpu')
    parser.add_argument(
        '--predict-step',
        type=int,
        default=8,
        help='run recognition every N frames and reuse the latest label '
        'between predictions')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.0,
        help='optional minimum confidence for changing label')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override config options, e.g. model.backbone.depth=18')
    parser.add_argument(
        '--normal-label',
        default='normal',
        help='label to show when prediction confidence is below threshold')
    parser.add_argument(
        '--show-score',
        action='store_true',
        help='also draw confidence score beside the label')
    return parser.parse_args()


def load_labels(label_file):
    with open(label_file, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f if line.strip()]
    if len(labels) != 2:
        raise ValueError(
            f'Expected exactly 2 labels, but got {len(labels)}: {labels}')
    return labels


def fix_swapped_video_and_label_args(args):
    video_suffixes = {'.avi', '.mkv', '.mov', '.mp4', '.webm'}
    video_path = Path(args.video)
    label_path = Path(args.label)

    looks_like_label = video_path.suffix.lower() in {'.txt', '.csv'}
    looks_like_video = label_path.suffix.lower() in video_suffixes
    if looks_like_label and looks_like_video:
        print('Warning: video and label arguments look swapped; using '
              f'{args.label} as video and {args.video} as label file.')
        args.video, args.label = args.label, args.video


def build_array_pipeline(cfg):
    """Convert a normal video test pipeline into an array-frame pipeline."""
    sample_cfg = None
    pipeline = []

    for step in cfg.test_pipeline:
        step_type = step['type']
        if step_type == 'SampleFrames':
            sample_cfg = step
            continue
        if step_type in EXCLUDED_PIPELINE_STEPS:
            continue
        pipeline.append(step)

    if sample_cfg is None:
        raise RuntimeError('Could not find SampleFrames in cfg.test_pipeline')

    pipeline.insert(0, dict(type='ArrayDecode'))
    return Compose(pipeline), sample_cfg


def get_window_size(sample_cfg):
    clip_len = int(sample_cfg.get('clip_len', 1))
    frame_interval = int(sample_cfg.get('frame_interval', 1))
    num_clips = int(sample_cfg.get('num_clips', 1))
    return max(1, clip_len * frame_interval * num_clips)


def validate_model_sampling(cfg, sample_cfg):
    backbone = cfg.model.get('backbone', {})
    if backbone.get('type') != 'ResNet3dSlowFast':
        return

    clip_len = int(sample_cfg.get('clip_len', 1))
    resample_rate = int(backbone.get('resample_rate', 8))
    if clip_len % resample_rate != 0:
        raise ValueError(
            'Invalid SlowFast test clip_len. '
            f'clip_len={clip_len} is not divisible by '
            f'resample_rate={resample_rate}. SlowFast splits each sampled '
            'clip into slow and fast pathways, and their temporal sizes must '
            'line up during lateral fusion. Use clip_len values like '
            f'{resample_rate}, {resample_rate * 2}, '
            f'{resample_rate * 3}, or {resample_rate * 4}. '
            'For this config, try clip_len=16 or clip_len=8.')


def sample_window_frames(frames, sample_cfg):
    clip_len = int(sample_cfg.get('clip_len', 1))
    frame_interval = int(sample_cfg.get('frame_interval', 1))
    num_clips = int(sample_cfg.get('num_clips', 1))
    num_sampled_frames = clip_len * num_clips

    frames = list(frames)
    needed = num_sampled_frames * frame_interval
    if len(frames) < needed:
        return None

    start = len(frames) - needed
    indices = start + np.arange(num_sampled_frames) * frame_interval
    return [frames[int(idx)] for idx in indices]


def synchronize_if_cuda(model):
    if next(model.parameters()).is_cuda:
        torch.cuda.synchronize()


def predict_label(model, frames, labels, test_pipeline, sample_cfg):
    sampled_frames = sample_window_frames(frames, sample_cfg)
    if sampled_frames is None:
        return None, None, None

    data = dict(
        array=sampled_frames,
        frame_inds=np.arange(len(sampled_frames)),
        total_frames=len(sampled_frames),
        num_clips=int(sample_cfg.get('num_clips', 1)),
        clip_len=int(sample_cfg.get('clip_len', len(sampled_frames))),
        img_shape=sampled_frames[0].shape[:2],
        modality='RGB',
        label=-1)

    synchronize_if_cuda(model)
    start_time = time.perf_counter()
    result = inference_recognizer(model, data, test_pipeline=test_pipeline)
    synchronize_if_cuda(model)
    infer_time = time.perf_counter() - start_time

    scores = result.pred_score.detach().cpu().numpy()
    label_idx = int(scores.argmax())
    return labels[label_idx], float(scores[label_idx]), infer_time


def draw_label(frame, label, score=None):
    color = (0, 0, 255) if label.lower() == 'fighting' else (0, 200, 0)
    text = label.upper()
    if score is not None:
        text = f'{text} {score:.2f}'

    (text_w, text_h), baseline = cv2.getTextSize(
        text, FONT_FACE, FONT_SCALE, FONT_THICKNESS)
    x, y = TEXT_ORIGIN
    box_tl = (x - BOX_PADDING, y - text_h - BOX_PADDING)
    box_br = (x + text_w + BOX_PADDING, y + baseline + BOX_PADDING)

    cv2.rectangle(frame, box_tl, box_br, (0, 0, 0), thickness=-1)
    cv2.putText(frame, text, TEXT_ORIGIN, FONT_FACE, FONT_SCALE, color,
                FONT_THICKNESS, cv2.LINE_AA)


def main():
    args = parse_args()
    fix_swapped_video_and_label_args(args)
    labels = load_labels(args.label)

    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)

    device = torch.device(args.device)
    model = init_recognizer(cfg, args.checkpoint, device=device)
    test_pipeline, sample_cfg = build_array_pipeline(model.cfg)
    validate_model_sampling(model.cfg, sample_cfg)
    window_size = get_window_size(sample_cfg)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {args.video}')

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_dir = Path(args.out_file).parent
    if str(out_dir) not in ('', '.'):
        out_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out_file, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f'Could not create output video: {args.out_file}')

    frame_buffer = deque(maxlen=window_size)
    current_label = args.normal_label
    current_score = None
    infer_times = []
    progress = mmengine.ProgressBar(num_frames)

    frame_idx = 0
    total_start_time = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb_frame = frame[:, :, ::-1]
        frame_buffer.append(rgb_frame)

        should_predict = (
            len(frame_buffer) == window_size
            and frame_idx % max(1, args.predict_step) == 0)

        if should_predict:
            label, score, infer_time = predict_label(model, frame_buffer,
                                                     labels, test_pipeline,
                                                     sample_cfg)
            if label is not None:
                infer_times.append(infer_time)
                if score >= args.threshold:
                    current_label = label
                    current_score = score
                else:
                    current_label = args.normal_label
                    current_score = score

        draw_label(frame, current_label,
                   current_score if args.show_score else None)
        writer.write(frame)

        frame_idx += 1
        progress.update()

    total_time = time.perf_counter() - total_start_time
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    processed_fps = frame_idx / total_time if total_time > 0 else 0.0
    print(f'\nOutput saved to: {args.out_file}')
    print('\nBenchmark summary')
    print(f'  Video frames processed: {frame_idx}')
    print(f'  Video FPS: {fps:.2f}')
    print(f'  Prediction window size: {window_size} frames')
    print(f'  Predict step: every {max(1, args.predict_step)} frames')
    print(f'  Total processing time: {total_time:.4f} s')
    print(f'  End-to-end processing speed: {processed_fps:.2f} FPS')

    if infer_times:
        infer_times_np = np.array(infer_times)
        print(f'  Number of model predictions: {len(infer_times)}')
        print(f'  Avg inference_recognizer time: '
              f'{infer_times_np.mean():.4f} s/window')
        print(f'  Min inference_recognizer time: '
              f'{infer_times_np.min():.4f} s/window')
        print(f'  Max inference_recognizer time: '
              f'{infer_times_np.max():.4f} s/window')
        print(f'  Prediction throughput: '
              f'{1.0 / infer_times_np.mean():.2f} windows/s')
    else:
        print('  Number of model predictions: 0')
        print('  No prediction was made. The video may be shorter than the '
              'model prediction window.')


if __name__ == '__main__':
    main()
