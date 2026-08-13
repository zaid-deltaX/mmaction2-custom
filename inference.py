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
        description='Run action recognition on a video, draw the predicted '
        'class on frames, and optionally show/save the result.')
    parser.add_argument('config', help='MMAction2 model config file')
    parser.add_argument('checkpoint', help='trained checkpoint file')
    parser.add_argument('video', help='input video file')
    parser.add_argument('label_map', help='label_map.txt file')
    parser.add_argument(
        '--out-file',
        default='outputs/inference_result.mp4',
        help='output video path')
    parser.add_argument(
        '--device', default='cuda:0', help='device, e.g. cuda:0 or cpu')
    parser.add_argument(
        '--predict-step',
        type=int,
        default=8,
        help='run recognition every N frames and reuse the latest label')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.0,
        help='minimum confidence for accepting a prediction')
    parser.add_argument(
        '--fallback-label',
        default=None,
        help='label shown when score is below threshold')
    parser.add_argument(
        '--show',
        action='store_true',
        help='display output frames in an OpenCV window')
    parser.add_argument(
        '--display-width',
        type=int,
        default=1280,
        help='maximum OpenCV preview window width when using --show')
    parser.add_argument(
        '--display-height',
        type=int,
        default=720,
        help='maximum OpenCV preview window height when using --show')
    parser.add_argument(
        '--show-score',
        action='store_true',
        help='draw confidence score beside the predicted class')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override config options, e.g. test_pipeline.1.num_clips=1')
    return parser.parse_args()


def load_labels(label_map):
    with open(label_map, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f if line.strip()]
    if not labels:
        raise RuntimeError(f'No labels found in {label_map}')
    return labels


def build_array_pipeline(cfg):
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
            f'Invalid SlowFast clip_len={clip_len}. It must be divisible by '
            f'resample_rate={resample_rate}. Try clip_len=8 or 16.')


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
    is_alert = label.lower() in {'fighting', 'fight', 'violence'}
    color = (0, 0, 255) if is_alert else (0, 200, 0)
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


def resize_for_display(frame, max_width, max_height):
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return frame

    display_size = (int(width * scale), int(height * scale))
    return cv2.resize(frame, display_size, interpolation=cv2.INTER_AREA)


def main():
    args = parse_args()
    labels = load_labels(args.label_map)

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
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_file), cv2.VideoWriter_fourcc(*'mp4v'), fps,
        (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f'Could not create output video: {out_file}')

    fallback_label = args.fallback_label or labels[0]
    current_label = fallback_label
    current_score = None
    frame_buffer = deque(maxlen=window_size)
    infer_times = []

    print('Inference setup')
    print(f'  Config: {args.config}')
    print(f'  Checkpoint: {args.checkpoint}')
    print(f'  Video: {args.video}')
    print(f'  Labels: {labels}')
    print(f'  Window size: {window_size} frames')
    print(f'  Predict step: {max(1, args.predict_step)} frames')

    progress = mmengine.ProgressBar(frame_count)
    frame_idx = 0
    total_start = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_buffer.append(frame[:, :, ::-1])
        should_predict = (
            len(frame_buffer) == window_size
            and frame_idx % max(1, args.predict_step) == 0)

        if should_predict:
            label, score, infer_time = predict_label(model, frame_buffer,
                                                     labels, test_pipeline,
                                                     sample_cfg)
            if label is not None:
                infer_times.append(infer_time)
                current_label = label if score >= args.threshold else fallback_label
                current_score = score

        draw_label(frame, current_label,
                   current_score if args.show_score else None)
        writer.write(frame)

        if args.show:
            display_frame = resize_for_display(
                frame, args.display_width, args.display_height)
            cv2.imshow('Action Recognition', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1
        progress.update()

    total_time = time.perf_counter() - total_start
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    processed_fps = frame_idx / total_time if total_time > 0 else 0.0
    print(f'\nOutput saved to: {out_file}')
    print('\nBenchmark summary')
    print(f'  Video frames processed: {frame_idx}')
    print(f'  Video FPS: {fps:.2f}')
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
    else:
        print('  Number of model predictions: 0')
        print('  No prediction was made. Reduce clip_len, frame_interval, or '
              'num_clips if the video is short.')


if __name__ == '__main__':
    main()
