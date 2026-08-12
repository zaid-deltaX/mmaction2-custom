# MMAction2 Practical Guide: Inference, Custom Training, Data, and Deployment

This repository is OpenMMLab's MMAction2: a PyTorch/MMEngine framework for video understanding. It contains model definitions, dataset classes, data transforms, training/testing entrypoints, demo scripts, and many ready-to-use configs for action recognition, skeleton recognition, spatio-temporal action detection, retrieval, localization, and multimodal video tasks.

This guide is written for the common practical goal: train a custom action recognition model for classes such as `fight`, `smoking`, `normal`, etc., then run inference on new videos.

## 1. What This Repo Contains

Important directories:

```text
mmaction/
  apis/                 High-level Python inference APIs.
  datasets/             Dataset classes: VideoDataset, RawframeDataset, AVADataset, PoseDataset, etc.
  datasets/transforms/  Video loading, frame sampling, resizing, cropping, formatting.
  models/               Backbones, heads, recognizers, losses, localizers, detection heads.
  evaluation/           Accuracy, AVA, ActivityNet, retrieval metrics.
  engine/               MMEngine hooks, optimizers, loops, runners.

configs/
  recognition/          Clip/video-level action recognition configs.
  detection/            Person-level spatio-temporal action detection configs.
  skeleton/             Pose/skeleton-based action recognition configs.
  recognition_audio/    Audio-based action recognition configs.
  retrieval/, multimodal/, localization/  Other video understanding tasks.

tools/
  train.py              Main training entrypoint.
  test.py               Main validation/test entrypoint.
  dist_train.sh         Multi-GPU training wrapper.
  dist_test.sh          Multi-GPU test wrapper.
  data/                 Dataset preparation helpers and label maps.
  deployment/           Export/serving helpers such as ONNX and TorchServe.
  visualizations/       Dataset browsing, scheduler visualization, GradCAM.

demo/
  demo.py               Single-video action recognition demo.
  demo_inferencer.py    Unified inferencer demo.
  long_video_demo.py    Sliding/window-style long video inference demo.
  webcam_demo.py        Webcam action recognition.
  demo_spatiotemporal_det.py  Person-level action detection demo.
```

## 2. Core Idea

MMAction2 is config-driven.

A config file describes:

- `model`: recognizer type, backbone, classification head, number of classes.
- `train_pipeline`, `val_pipeline`, `test_pipeline`: how video is decoded, sampled, augmented, formatted, and packed.
- `train_dataloader`, `val_dataloader`, `test_dataloader`: dataset path, annotation file, batch size, workers.
- `optim_wrapper`, `param_scheduler`: optimizer and learning-rate schedule.
- `train_cfg`, `val_cfg`, `test_cfg`: training and evaluation loops.
- `load_from`: checkpoint to initialize from.
- `work_dir`: where logs and checkpoints are saved.

`tools/train.py` loads the config with `mmengine.Config.fromfile`, builds an MMEngine `Runner`, and calls `runner.train()`.

`tools/test.py` loads the config and checkpoint, builds the same runner, and calls `runner.test()`.

`demo/demo.py` and `mmaction/apis/inference.py` build a model from the config, load a checkpoint, run the `test_pipeline`, and call `model.test_step(...)`.

## 3. Task Choice for Fight/Smoking Use Cases

There are two different problem definitions:

### A. Clip-Level Action Recognition

Use this when each short video clip has one main label:

```text
clip_001.mp4 -> fight
clip_002.mp4 -> smoking
clip_003.mp4 -> normal
```

This is the easiest path. Use `VideoDataset` or `RawframeDataset`.

Good starting models:

- TSN: fast, simple baseline, good first experiment.
- TSM: efficient temporal model.
- SlowFast / SlowOnly: stronger spatio-temporal models.
- VideoMAE / Swin / UniFormer: stronger transformer-style models, usually heavier.

For a first custom model, start with TSN or TSM. After your data pipeline works, try SlowFast or VideoMAE.

### B. Spatio-Temporal Action Detection

Use this when you need to know which person is doing an action and where/when it happens:

```text
At timestamp 00:12, person bbox [x1,y1,x2,y2] -> smoking
At timestamp 00:38, person bbox [x1,y1,x2,y2] -> fighting
```

This uses AVA-style annotations and configs under `configs/detection/`. It usually needs:

- person bounding boxes,
- action labels per person per timestamp,
- often a separate person detector from MMDetection.

This is more work but better for CCTV or multi-person videos where only one person may be smoking/fighting.

## 4. Installation

Create an environment and install the OpenMMLab stack:

```bash
conda create -n mmaction2 python=3.8 -y
conda activate mmaction2

# Install PyTorch matching your CUDA/CPU environment first.
# See https://pytorch.org/get-started/locally/

pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet
mim install mmpose

pip install -v -e .
```

`mmdet` and `mmpose` are optional for simple RGB action recognition, but useful for spatio-temporal detection and skeleton demos.

## 5. Run Inference With an Existing Checkpoint

### Option 1: Unified Inferencer

```bash
python demo/demo_inferencer.py demo/demo.mp4 \
  --rec tsn \
  --print-result \
  --label-file tools/data/kinetics/label_map_k400.txt
```

This uses a known model name from the model metadata when available.

### Option 2: Explicit Config + Checkpoint

```bash
python demo/demo.py \
  configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py \
  https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth \
  demo/demo.mp4 \
  tools/data/kinetics/label_map_k400.txt \
  --device cuda:0
```

Use `--device cpu` if you do not have a GPU.

To save a visualized output video:

```bash
python demo/demo.py CONFIG CHECKPOINT input.mp4 label_map.txt \
  --out-filename demo/result.mp4
```

### Python API

```python
from operator import itemgetter
from mmaction.apis import init_recognizer, inference_recognizer

config = 'configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py'
checkpoint = 'path_or_url_to_checkpoint.pth'
video = 'my_video.mp4'
label_file = 'tools/data/kinetics/label_map_k400.txt'

model = init_recognizer(config, checkpoint, device='cuda:0')
result = inference_recognizer(model, video)

scores = result.pred_score.tolist()
labels = [line.strip() for line in open(label_file)]
top5 = sorted(enumerate(scores), key=itemgetter(1), reverse=True)[:5]

for class_id, score in top5:
    print(labels[class_id], score)
```

## 6. Custom Dataset Format for Clip-Level Recognition

For fighting/smoking classification, the simplest dataset layout is:

```text
data/my_actions/
  videos_train/
    fight_001.mp4
    fight_002.mp4
    smoking_001.mp4
    normal_001.mp4
  videos_val/
    fight_101.mp4
    smoking_101.mp4
    normal_101.mp4
  train.txt
  val.txt
  label_map.txt
```

`label_map.txt`:

```text
normal
fight
smoking
```

Labels are zero-based. In this example:

```text
normal  -> 0
fight   -> 1
smoking -> 2
```

`train.txt` for `VideoDataset`:

```text
fight_001.mp4 1
fight_002.mp4 1
smoking_001.mp4 2
normal_001.mp4 0
```

`val.txt`:

```text
fight_101.mp4 1
smoking_101.mp4 2
normal_101.mp4 0
```

Each line is:

```text
relative_video_path label_id
```

The path is relative to `data_prefix=dict(video=data_root)` in the config.

### Raw Frame Alternative

If you extract frames first, use `RawframeDataset`. The annotation format is:

```text
frame_directory total_frames label_id
```

Example:

```text
fight_001 163 1
smoking_001 122 2
normal_001 258 0
```

For most new projects, start with `VideoDataset`; it stores less data and is simpler.

## 7. How to Label/Annotate Your Dataset

For clip-level recognition:

1. Decide the exact classes.
2. Split long videos into short clips where one label is dominant.
3. Put clips into train/val/test folders.
4. Create `label_map.txt`.
5. Create `train.txt`, `val.txt`, and optionally `test.txt`.

Recommended clip strategy:

- Use clips of 2-10 seconds for visible actions like fighting or smoking.
- Avoid clips where multiple labels are equally true unless you intentionally model multi-label classification.
- Include a strong `normal` or `background` class.
- Include negative examples that look similar but are not the action, such as hugging/play-fighting for fight detection, or holding a phone/pen near the mouth for smoking.
- Keep train/val/test split person-wise or scene-wise when possible, so the model does not simply memorize the same person/place.

Annotation can be done with simple folder organization, spreadsheets, CVAT, Label Studio, Roboflow, or any tool that exports a CSV. For clip-level classification, you only need one label per clip; after export, convert it to MMAction2's text format.

Example CSV:

```csv
filename,label
fight_001.mp4,fight
smoking_001.mp4,smoking
normal_001.mp4,normal
```

Convert to `train.txt` using your class mapping:

```text
fight_001.mp4 1
smoking_001.mp4 2
normal_001.mp4 0
```

For person-level detection, use CVAT/Label Studio-style box annotation and convert to AVA CSV format:

```text
video_id,timestamp,x1,y1,x2,y2,label,entity_id
```

Coordinates are normalized to `[0, 1]`. See `tools/data/ava/AVA_annotation_explained.md`.

## 8. Create a Custom Training Config

Start from an existing recognition config. For a simple baseline:

```text
configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py
```

Create a new file, for example:

```text
configs/recognition/tsn/tsn_r50_custom_fight_smoking.py
```

Minimal custom config:

```python
_base_ = ['tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py']

dataset_type = 'VideoDataset'
data_root = 'data/my_actions/videos_train'
data_root_val = 'data/my_actions/videos_val'
ann_file_train = 'data/my_actions/train.txt'
ann_file_val = 'data/my_actions/val.txt'
ann_file_test = 'data/my_actions/val.txt'

model = dict(
    cls_head=dict(
        num_classes=3,
        average_clips='prob'))

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root)))

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        test_mode=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        test_mode=True))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_begin=1, val_interval=1)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,
        by_epoch=True,
        milestones=[20, 40],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=3, save_best='auto'))

load_from = 'https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth'
```

Important edits:

- `num_classes`: number of your classes.
- `data_root`, `data_root_val`: video folders.
- `ann_file_train`, `ann_file_val`, `ann_file_test`: annotation text files.
- `load_from`: pretrained checkpoint used to initialize training.
- `batch_size`: reduce if you run out of GPU memory.
- `max_epochs` and learning rate: tune for your dataset size.

## 9. Train

Single GPU:

```bash
python tools/train.py configs/recognition/tsn/tsn_r50_custom_fight_smoking.py \
  --work-dir work_dirs/tsn_r50_custom_fight_smoking
```

CPU only:

```bash
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/recognition/tsn/tsn_r50_custom_fight_smoking.py
```

Multi-GPU:

```bash
bash tools/dist_train.sh configs/recognition/tsn/tsn_r50_custom_fight_smoking.py 4
```

Resume:

```bash
python tools/train.py configs/recognition/tsn/tsn_r50_custom_fight_smoking.py \
  --resume work_dirs/tsn_r50_custom_fight_smoking/latest.pth
```

Use AMP:

```bash
python tools/train.py configs/recognition/tsn/tsn_r50_custom_fight_smoking.py --amp
```

Outputs go to `work_dirs/...`:

```text
work_dirs/tsn_r50_custom_fight_smoking/
  epoch_1.pth
  epoch_2.pth
  best_*.pth
  latest.pth
  TIMESTAMP/
    logs
    config copy
```

## 10. Test/Evaluate

Replace `best_xxx.pth` with the actual best checkpoint file saved in your work directory.

```bash
python tools/test.py \
  configs/recognition/tsn/tsn_r50_custom_fight_smoking.py \
  work_dirs/tsn_r50_custom_fight_smoking/best_xxx.pth
```

Dump predictions:

```bash
python tools/test.py CONFIG CHECKPOINT --dump predictions.pkl
```

Common metric for single-label recognition is top-1/top-5 accuracy via `AccMetric`.

For imbalanced classes such as rare fights, also inspect per-class precision/recall/F1 outside MMAction2 using dumped predictions or a custom evaluator. Accuracy alone can look good while missing rare events.

## 11. Inference With Your Custom Model

Create a custom label file:

```text
data/my_actions/label_map.txt
```

Example:

```text
normal
fight
smoking
```

Run:

```bash
python demo/demo.py \
  configs/recognition/tsn/tsn_r50_custom_fight_smoking.py \
  work_dirs/tsn_r50_custom_fight_smoking/best_xxx.pth \
  path/to/new_video.mp4 \
  data/my_actions/label_map.txt \
  --device cuda:0
```

For a long video, use:

```bash
python demo/long_video_demo.py \
  configs/recognition/tsn/tsn_r50_custom_fight_smoking.py \
  work_dirs/tsn_r50_custom_fight_smoking/best_xxx.pth \
  path/to/long_video.mp4 \
  data/my_actions/label_map.txt \
  outputs/long_video_result.mp4 \
  --device cuda:0 \
  --threshold 0.5
```

For webcam:

```bash
python demo/webcam_demo.py \
  configs/recognition/tsn/tsn_r50_custom_fight_smoking.py \
  work_dirs/tsn_r50_custom_fight_smoking/best_xxx.pth \
  data/my_actions/label_map.txt \
  --device cuda:0 \
  --threshold 0.5 \
  --average-size 5
```

## 12. What Model Format Is Produced?

Training produces PyTorch checkpoint files:

```text
.pth
```

A `.pth` checkpoint generally contains model weights plus metadata and optimizer/training state depending on how it was saved.

The model architecture is not fully contained in the `.pth` in a standalone deployment sense. You normally need both:

```text
config.py + checkpoint.pth
```

The config defines the model structure and preprocessing pipeline. The checkpoint provides learned weights.

For deployment, MMAction2 also has tooling under `tools/deployment/` for formats such as:

- ONNX export for some model families/tasks.
- TorchServe packaging via `mmaction2torchserve.py`.
- Published checkpoints via `publish_model.py`.

For production inference, common choices are:

- PyTorch runtime: easiest, uses MMAction2 APIs directly.
- ONNX Runtime: useful for cross-platform or optimized CPU/GPU serving when your model exports cleanly.
- TensorRT: possible through the OpenMMLab deployment ecosystem, best for NVIDIA production optimization, but requires extra validation.
- TorchServe: model server packaging for PyTorch deployments.

## 13. What Is the Inference Engine?

Inside this repo, default inference is PyTorch plus MMEngine/MMCV data transforms.

Flow:

```text
video path
  -> test_pipeline
     -> DecordInit / PyAVInit
     -> SampleFrames
     -> DecordDecode / PyAVDecode
     -> Resize/Crop
     -> FormatShape
     -> PackActionInputs
  -> recognizer model
     -> backbone
     -> classification head
  -> ActionDataSample.pred_score
```

The high-level API is:

```python
model = init_recognizer(config, checkpoint, device='cuda:0')
result = inference_recognizer(model, video_path)
```

The actual neural-network runtime is PyTorch unless you export to ONNX or another deployment backend.

## 14. How the Main Recognition Models Differ

Short version:

- TSN: samples sparse frames; fast and simple; good baseline.
- TSM: TSN-like but shifts temporal features; efficient video temporal modeling.
- I3D/C3D/R2Plus1D: 3D convolutional models; heavier, stronger temporal modeling.
- SlowOnly/SlowFast: strong video models; good for motion-heavy actions like fighting.
- X3D: efficient 3D model family.
- Swin/VideoMAE/TimeSformer/UniFormer: transformer-based or modern video backbones; often stronger but heavier.
- PoseC3D/STGCN/AGCN: skeleton/keypoint-based; useful when action is body-motion driven and background should matter less.

For fighting:

- Start with RGB `TSM` or `SlowFast`.
- If camera/background changes a lot, try skeleton-based recognition.
- If you need person-specific boxes, move to AVA-style spatio-temporal detection.

For smoking:

- RGB clip-level recognition may work if the cigarette/action is visible.
- Person/object-level detection may be better if the signal is small.
- Include hard negatives: eating, drinking, touching face, phone near mouth.

## 15. Data Quality Checklist

Before training seriously:

- Every class has enough examples.
- Train/val/test split does not leak the same source video across splits.
- Clips are trimmed so labels are actually visible.
- `normal` class is diverse.
- Label ids match `label_map.txt`.
- Videos are readable by Decord/PyAV.
- Class imbalance is handled by collecting more data, re-sampling, or weighting.
- Validation set reflects real deployment footage.

Use the dataset browser to inspect transformed samples:

```bash
python tools/visualizations/browse_dataset.py \
  configs/recognition/tsn/tsn_r50_custom_fight_smoking.py \
  browse_out \
  --mode pipeline
```

## 16. Common Problems

### Out of GPU Memory

Reduce:

- `train_dataloader.batch_size`
- `clip_len`
- `num_clips`
- input crop size

Or use:

```bash
python tools/train.py CONFIG --amp
```

### Wrong Labels

Check:

- `num_classes`
- `label_map.txt`
- annotation ids
- class order

### Poor Validation Accuracy

Check:

- Is the action visible in the clip?
- Is the clip too long with too much irrelevant time?
- Are classes imbalanced?
- Are train and validation distributions too different?
- Is the model too weak or too heavy for the dataset size?

### Good Validation, Bad Real Video

Usually this is data mismatch. Add real deployment-like examples, harder negatives, camera angles, lighting, compression artifacts, and backgrounds.

## 17. Recommended First Experiment

For your use case, do this first:

1. Define classes:

```text
normal
fight
smoking
```

2. Collect 100-500 short clips per class if possible.
3. Split by source video/person/location.
4. Create `data/my_actions/train.txt`, `val.txt`, `label_map.txt`.
5. Fine-tune TSN or TSM from Kinetics pretrained weights.
6. Test on held-out videos.
7. Run `long_video_demo.py` on real long footage.
8. Inspect false positives/false negatives.
9. Add hard examples and retrain.
10. Try SlowFast if TSN/TSM is not enough.

## 18. Minimal End-to-End Command Sequence

```bash
# 1. Train
python tools/train.py configs/recognition/tsn/tsn_r50_custom_fight_smoking.py \
  --work-dir work_dirs/tsn_r50_custom_fight_smoking

# 2. Test
python tools/test.py configs/recognition/tsn/tsn_r50_custom_fight_smoking.py \
  work_dirs/tsn_r50_custom_fight_smoking/best_xxx.pth

# 3. Single video inference
python demo/demo.py configs/recognition/tsn/tsn_r50_custom_fight_smoking.py \
  work_dirs/tsn_r50_custom_fight_smoking/best_xxx.pth \
  path/to/video.mp4 \
  data/my_actions/label_map.txt \
  --device cuda:0

# 4. Long video inference
python demo/long_video_demo.py configs/recognition/tsn/tsn_r50_custom_fight_smoking.py \
  work_dirs/tsn_r50_custom_fight_smoking/best_xxx.pth \
  path/to/long_video.mp4 \
  data/my_actions/label_map.txt \
  outputs/result.mp4 \
  --device cuda:0 \
  --threshold 0.5
```

## 19. Where to Read More in This Repo

- Inference guide: `docs/en/user_guides/inference.md`
- Fine-tuning guide: `docs/en/user_guides/finetune.md`
- Dataset preparation: `docs/en/user_guides/prepare_dataset.md`
- Training/testing: `docs/en/user_guides/train_test.md`
- Framework walkthrough: `docs/en/get_started/guide_to_framework.md`
- Demo commands: `demo/README.md`
- Dataset customization: `docs/en/advanced_guides/customize_dataset.md`
- AVA annotations: `tools/data/ava/AVA_annotation_explained.md`
- Recognition configs: `configs/recognition/`
- Detection configs: `configs/detection/`
- Skeleton configs: `configs/skeleton/`

## 20. Mental Model to Keep

For custom clip-level action recognition, MMAction2 needs only four things:

```text
1. videos
2. annotation txt files
3. config.py with correct paths and num_classes
4. pretrained .pth checkpoint to fine-tune from
```

At inference time it needs:

```text
1. config.py
2. trained checkpoint.pth
3. input video
4. label_map.txt
```

That is the whole loop.
