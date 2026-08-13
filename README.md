# Custom Action Recognition: Training and Inference

This repository is set up to train and run video action recognition models with MMAction2.

For repository setup, dependency installation, and the original MMAction2 environment instructions, refer to [README_old.md](README_old.md).

The current custom task is:

```text
fighting
normal
```

The root-level scripts are:

```text
train.py      Train a selected MMAction2 config on a folder dataset.
inference.py  Run a trained model on a video and draw predictions on frames.
```

## 1. Dataset Format

Prepare your dataset like this:

```text
dataset/v2/
  train/
    fighting/
      fight_001.mp4
      fight_002.mp4
    normal/
      normal_001.mp4
      normal_002.mp4
  val/
    fighting/
      fight_101.mp4
    normal/
      normal_101.mp4
  test/
    test_1.mp4
```

The folder names are the class names.

When you run `train.py`, it automatically creates:

```text
dataset/v2/label_map.txt
dataset/v2/train.txt
dataset/v2/val.txt
```

Example `label_map.txt`:

```text
fighting
normal
```

Example `train.txt`:

```text
fighting/fight_001.mp4 0
normal/normal_001.mp4 1
```

Important: the order in `label_map.txt` must match the class indices in `train.txt` and `val.txt`. The training script handles this if you pass `--classes` explicitly.

## 2. Available Model Configs

created these config files for the models. You can create your own from configs/recognition/ folder .Use one of these config files when training:

```text
configs/recognition/tsn/tsn_r50_fighting_normal.py
configs/recognition/tsm/tsm_r50_fighting_normal.py
configs/recognition/slowfast/slowfast_r50_8xb8-fighting_normal.py
configs/recognition/swin/swin-tiny-p244_fighting_normal.py
configs/recognition/videomae/vit-videomae_p16_fighting_normal.py
```


## 3. Train a Model

Dataset path can be handled in two ways:

```text
1. Put dataset paths directly inside the config file.
2. Pass the dataset root to train.py as an argument.
```

The recommended workflow in this repo is option 2. When you pass `dataset/v2` to `train.py`, the script automatically overrides the config dataset paths for training, validation, and testing.

Basic command:

```powershell
python train.py `
  configs/recognition/tsm/tsm_r50_fighting_normal.py `
  dataset/v2 `
  --classes fighting normal `
  --work-dir work_dirs/tsm_r50_fighting_normal
```

In that command:

```text
configs/recognition/tsm/tsm_r50_fighting_normal.py  -> model/config file
dataset/v2                                           -> dataset root argument
```

`train.py` uses the dataset argument to set:

```text
train data root -> dataset/v2/train
val data root   -> dataset/v2/val
train ann file  -> dataset/v2/train.txt
val ann file    -> dataset/v2/val.txt
test ann file   -> dataset/v2/val.txt
```

Train TSN:

```powershell
python train.py `
  configs/recognition/tsn/tsn_r50_fighting_normal.py `
  dataset/v2 `
  --classes fighting normal `
  --work-dir work_dirs/tsn_r50_fighting_normal
```

Train TSM:

```powershell
python train.py `
  configs/recognition/tsm/tsm_r50_fighting_normal.py `
  dataset/v2 `
  --classes fighting normal `
  --work-dir work_dirs/tsm_r50_fighting_normal
```

Train SlowFast:

```powershell
python train.py `
  configs/recognition/slowfast/slowfast_r50_8xb8-fighting_normal.py `
  dataset/v2 `
  --classes fighting normal `
  --work-dir work_dirs/slowfast_r50_fighting_normal
```

Train Swin:

```powershell
python train.py `
  configs/recognition/swin/swin-tiny-p244_fighting_normal.py `
  dataset/v2 `
  --classes fighting normal `
  --work-dir work_dirs/swin_tiny_p244_fighting_normal
```

Train VideoMAE:

```powershell
python train.py `
  configs/recognition/videomae/vit-videomae_p16_fighting_normal.py `
  dataset/v2 `
  --classes fighting normal `
  --work-dir work_dirs/videomae_fighting_normal `
  --amp
```

Useful training options:

```powershell
--max-epochs 50
--batch-size 2
--num-workers 2
--checkpoint-interval 1
--amp
--resume
```

Example with overrides:

```powershell
python train.py `
  configs/recognition/slowfast/slowfast_r50_8xb8-fighting_normal.py `
  dataset/v2 `
  --classes fighting normal `
  --work-dir work_dirs/slowfast_r50_fighting_normal `
  --max-epochs 50 `
  --batch-size 2 `
  --num-workers 1 `
  --checkpoint-interval 4
```

Checkpoints are saved inside the selected `--work-dir`, for example:

```text
work_dirs/slowfast_r50_fighting_normal/epoch_50.pth
```

## 4. Run Inference on a Video

Basic command:

```powershell
python inference.py `
  configs/recognition/slowfast/slowfast_r50_8xb8-fighting_normal.py `
  work_dirs/slowfast_r50_fighting_normal/epoch_50.pth `
  dataset/v2/test/test_1.mp4 `
  dataset/v2/label_map.txt `
  --out-file outputs/slowfast_output/output_1.mp4 `
  --device cuda:0 `
  --show-score
```

To show the result live in an OpenCV window:

```powershell
python inference.py `
  configs/recognition/tsm/tsm_r50_fighting_normal.py `
  work_dirs/tsm_r50_fighting_normal/epoch_50.pth `
  dataset/v2/test/test_1.mp4 `
  dataset/v2/label_map.txt `
  --out-file outputs/tsm_output/output_1.mp4 `
  --device cuda:0 `
  --show `
  --show-score
```

Press `q` to stop the OpenCV window.

## 5. Inference Settings

Action recognition is usually window-based, not frame-by-frame.

A model prediction is made from a video window:

```text
window = clip_len * frame_interval * num_clips
```

Example:

```python
clip_len = 32
frame_interval = 2
num_clips = 1
```

The model receives 32 sampled frames, taken every 2 frames, covering:

```text
32 * 2 * 1 = 64 original video frames
```

At 30 FPS, that is about:

```text
64 / 30 = 2.13 seconds
```

For short videos or faster response, reduce the test sampling:

```powershell
--cfg-options test_pipeline.1.clip_len=16 test_pipeline.1.frame_interval=2 test_pipeline.1.num_clips=1
```

SlowFast note: `clip_len` should be divisible by the SlowFast `resample_rate`, usually `8`. Safe values are usually:

```text
8, 16, 24, 32
```

For SlowFast short-video inference, this is a good starting point:

```powershell
--cfg-options test_pipeline.1.clip_len=16 test_pipeline.1.frame_interval=2 test_pipeline.1.num_clips=1
```

## 6. Benchmark Models

`inference.py` prints timing after each run:

```text
Benchmark summary
  Video frames processed: ...
  Video FPS: ...
  Total processing time: ...
  End-to-end processing speed: ...
  Number of model predictions: ...
  Avg inference_recognizer time: ... s/window
```

Use this number to compare model inference speed:

```text
Avg inference_recognizer time
```

That is the average time for one model prediction window.

Use this number to estimate practical application speed:

```text
End-to-end processing speed
```

That includes video reading, preprocessing, model inference, drawing text, and writing the output video.

For fair benchmarking, use the same:

```text
same input video
same device
same test_pipeline clip_len/frame_interval/num_clips
same predict-step
same output settings
```

Example benchmark command:

```powershell
python inference.py `
  CONFIG.py `
  CHECKPOINT.pth `
  dataset/v2/test/test_1.mp4 `
  dataset/v2/label_map.txt `
  --out-file outputs/benchmark/output.mp4 `
  --device cuda:0 `
  --predict-step 8 `
  --show-score
```

## 7. Common Problems

If labels are swapped, check:

```text
dataset/v2/label_map.txt
dataset/v2/train.txt
dataset/v2/val.txt
```

The label map order must match the numeric labels.

If no prediction is made on a short video, reduce:

```text
clip_len
frame_interval
num_clips
```

If CUDA runs out of memory, try:

```powershell
--batch-size 1
--amp
```

For inference, try:

```powershell
--cfg-options test_pipeline.1.num_clips=1
```
