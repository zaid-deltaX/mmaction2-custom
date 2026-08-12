# Train a Custom Fighting/Smoking VideoMAE Model

This guide trains a transformer-based **VideoMAE ViT-Base** recognizer on the same custom dataset used for your TSN experiment:

- `fighting`
- `smoking`

Current local dataset root:

```text
C:\Projects\action_recognition\mmaction2\dataset\v1
```

Recommended custom config:

```text
configs/recognition/videomae/vit-videomae_p16_fighting_smoking.py
```

VideoMAE is much heavier than TSN. Start with a small batch size and use AMP if GPU memory is tight.

## 1. Dataset Layout

Expected folders:

```text
C:\Projects\action_recognition\mmaction2\dataset\v1\
  train\
    fighting\
    smoking\
  val\
    fighting\
    smoking\
  label_map.txt
  train.txt
  val.txt
```

Class mapping:

```text
fighting -> 0
smoking  -> 1
```

Your `label_map.txt` should contain:

```text
fighting
smoking
```

Your annotation files should contain relative video paths plus labels:

```text
fighting/fight_001.mp4 0
smoking/smoke_001.mp4 1
```

For the current binary dataset, expected counts are:

```text
train.txt: 60
val.txt:   20
```

## 2. Regenerate Labels If Needed

Run this from PowerShell in the MMAction2 repo root:

```powershell
cd C:\Projects\action_recognition\mmaction2
$DATASET_ROOT = "C:\Projects\action_recognition\mmaction2\dataset\v1"

"fighting", "smoking" | Set-Content "$DATASET_ROOT\label_map.txt"

Get-ChildItem "$DATASET_ROOT\train\fighting" -File |
  Sort-Object Name |
  ForEach-Object { "fighting/$($_.Name) 0" } |
  Set-Content "$DATASET_ROOT\train.txt"

Get-ChildItem "$DATASET_ROOT\train\smoking" -File |
  Sort-Object Name |
  ForEach-Object { "smoking/$($_.Name) 1" } |
  Add-Content "$DATASET_ROOT\train.txt"

Get-ChildItem "$DATASET_ROOT\val\fighting" -File |
  Sort-Object Name |
  ForEach-Object { "fighting/$($_.Name) 0" } |
  Set-Content "$DATASET_ROOT\val.txt"

Get-ChildItem "$DATASET_ROOT\val\smoking" -File |
  Sort-Object Name |
  ForEach-Object { "smoking/$($_.Name) 1" } |
  Add-Content "$DATASET_ROOT\val.txt"
```

Check:

```powershell
Get-Content "$DATASET_ROOT\label_map.txt"
Get-Content "$DATASET_ROOT\train.txt" -TotalCount 5
Get-Content "$DATASET_ROOT\val.txt" -TotalCount 5
(Get-Content "$DATASET_ROOT\train.txt").Count
(Get-Content "$DATASET_ROOT\val.txt").Count
```

## 3. Prepare the VideoMAE Config

The stock `configs/recognition/videomae/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400.py` config in this project is mainly a Kinetics-400 test config.

For custom training, make sure `configs/recognition/videomae/vit-videomae_p16_fighting_smoking.py` contains a full train/val/test setup like this:

```python
_base_ = ['../../_base_/default_runtime.py']

checkpoint = (
    'https://download.openmmlab.com/mmaction/v1.0/recognition/videomae/'
    'vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth'
)

model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='VisionTransformer',
        img_size=224,
        patch_size=16,
        embed_dims=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        num_frames=16,
        norm_cfg=dict(type='LN', eps=1e-6),
        drop_path_rate=0.1),
    cls_head=dict(
        type='TimeSformerHead',
        num_classes=2,
        in_channels=768,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'))

dataset_type = 'VideoDataset'
data_root = r'C:\Projects\action_recognition\mmaction2\dataset\v1\train'
data_root_val = r'C:\Projects\action_recognition\mmaction2\dataset\v1\val'
ann_file_train = r'C:\Projects\action_recognition\mmaction2\dataset\v1\train.txt'
ann_file_val = r'C:\Projects\action_recognition\mmaction2\dataset\v1\val.txt'
ann_file_test = r'C:\Projects\action_recognition\mmaction2\dataset\v1\val.txt'

file_client_args = dict(io_backend='disk')

train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=16, frame_interval=4, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=4,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=4,
        num_clips=5,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,
    val_begin=1,
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

base_lr = 1e-5
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=base_lr,
        betas=(0.9, 0.999),
        weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
    clip_grad=dict(max_norm=1, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=45,
        eta_min=base_lr / 100,
        by_epoch=True,
        begin=5,
        end=50,
        convert_to_iter_based=True)
]

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=3, save_best='auto'),
    logger=dict(interval=10))

load_from = checkpoint
auto_scale_lr = dict(enable=False, base_batch_size=16)
```

Important lines:

```python
num_classes=2
batch_size=2
load_from = checkpoint
```

`load_from` initializes from the Kinetics-400 VideoMAE checkpoint. The final classification layer is different because your dataset has 2 classes instead of 400, so MMAction2 will initialize the new head for your custom task.

If you train on Linux instead of Windows, replace the dataset paths with:

```python
data_root = '/home/mohammad/projects/mmaction2/dataset/v1/train'
data_root_val = '/home/mohammad/projects/mmaction2/dataset/v1/val'
ann_file_train = '/home/mohammad/projects/mmaction2/dataset/v1/train.txt'
ann_file_val = '/home/mohammad/projects/mmaction2/dataset/v1/val.txt'
ann_file_test = '/home/mohammad/projects/mmaction2/dataset/v1/val.txt'
```

## 4. Verify Video Paths

Run from PowerShell:

```powershell
$DATASET_ROOT = "C:\Projects\action_recognition\mmaction2\dataset\v1"

Get-Content "$DATASET_ROOT\train.txt" | ForEach-Object {
  $parts = $_ -split "\s+"
  $video = $parts[0]
  if (-not (Test-Path "$DATASET_ROOT\train\$video")) {
    "Missing train file: $video"
  }
}

Get-Content "$DATASET_ROOT\val.txt" | ForEach-Object {
  $parts = $_ -split "\s+"
  $video = $parts[0]
  if (-not (Test-Path "$DATASET_ROOT\val\$video")) {
    "Missing val file: $video"
  }
}
```

If nothing prints, paths are good.

## 5. Browse Samples Before Training

```powershell
python tools/visualizations/browse_dataset.py `
  configs/recognition/videomae/vit-videomae_p16_fighting_smoking.py `
  browse_out_videomae_fighting_smoking `
  --mode pipeline `
  --show-number 6
```

Open the generated images and confirm:

- fighting videos show fighting.
- smoking videos show smoking.
- labels are not swapped.
- crops still contain the important action.

## 6. Train VideoMAE

Basic training:

```powershell
python tools/train.py `
  configs/recognition/videomae/vit-videomae_p16_fighting_smoking.py `
  --work-dir work_dirs/videomae_fighting_smoking
```

Recommended first attempt with AMP:

```powershell
python tools/train.py `
  configs/recognition/videomae/vit-videomae_p16_fighting_smoking.py `
  --work-dir work_dirs/videomae_fighting_smoking `
  --amp
```

If CUDA runs out of memory:

1. Lower `train_dataloader.batch_size` from `2` to `1`.
2. Keep `--amp`.
3. Lower `num_workers` to `2` if dataloading is unstable on Windows.

Resume training:

```powershell
python tools/train.py `
  configs/recognition/videomae/vit-videomae_p16_fighting_smoking.py `
  --work-dir work_dirs/videomae_fighting_smoking `
  --resume work_dirs/videomae_fighting_smoking/latest.pth `
  --amp
```

## 7. Test the Best Checkpoint

Replace `best_xxx.pth` with the actual best checkpoint name in `work_dirs/videomae_fighting_smoking`.

```powershell
python tools/test.py `
  configs/recognition/videomae/vit-videomae_p16_fighting_smoking.py `
  work_dirs/videomae_fighting_smoking/best_xxx.pth
```

Expected metric:

```text
acc/top1
```

Because the dataset is very small, validation accuracy can jump around. Do not trust one epoch blindly; inspect failed examples if possible.

## 8. Inference

Single video:

```powershell
python demo/demo.py `
  configs/recognition/videomae/vit-videomae_p16_fighting_smoking.py `
  work_dirs/videomae_fighting_smoking/best_xxx.pth `
  C:\path\to\new_video.mp4 `
  C:\Projects\action_recognition\mmaction2\dataset\v1\label_map.txt `
  --device cuda:0
```

CPU inference:

```powershell
python demo/demo.py `
  configs/recognition/videomae/vit-videomae_p16_fighting_smoking.py `
  work_dirs/videomae_fighting_smoking/best_xxx.pth `
  C:\path\to\new_video.mp4 `
  C:\Projects\action_recognition\mmaction2\dataset\v1\label_map.txt `
  --device cpu
```

Long video:

```powershell
python demo/long_video_demo.py `
  configs/recognition/videomae/vit-videomae_p16_fighting_smoking.py `
  work_dirs/videomae_fighting_smoking/best_xxx.pth `
  C:\path\to\long_video.mp4 `
  C:\Projects\action_recognition\mmaction2\dataset\v1\label_map.txt `
  outputs\videomae_long_video_result.mp4 `
  --device cuda:0 `
  --threshold 0.5
```

## 9. Notes for This Dataset

- Binary VideoMAE will always choose either `fighting` or `smoking`, even for a normal video.
- Add a `normal` class later if you need real-world deployment behavior.
- With only 60 training clips, VideoMAE can overfit quickly. Watch validation accuracy and loss.
- If validation accuracy is unstable, try `max_epochs=20` or `30` first.
- If training is too slow, TSN is expected to be much faster; VideoMAE trades speed for stronger temporal transformer features.
- Always regenerate `train.txt` and `val.txt` after adding, deleting, or renaming videos.

## 10. Optional 3-Class Setup Later

If you add normal videos later:

```text
normal   -> 0
fighting -> 1
smoking  -> 2
```

Then update:

```python
model = dict(
    cls_head=dict(
        num_classes=3,
        average_clips='prob'))
```

Also regenerate `label_map.txt`, `train.txt`, and `val.txt` so normal clips are listed.
