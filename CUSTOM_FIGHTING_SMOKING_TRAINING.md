# Train a Custom Fighting/Smoking Action Recognition Model

This guide gives two dataset/training setups:

- **Current setup, binary:** `fighting` vs `smoking`
- **Optional later setup, 3-class:** `normal` vs `fighting` vs `smoking`

Your current dataset root is:

```bash
DATASET_ROOT=/home/mohammad/projects/mmaction2/dataset/v1
```

Your current config file is:

```text
configs/recognition/tsn/tsn_r50_fighting_smoking.py
```

## A. Binary Setup: Fighting vs Smoking

Use this setup now if you do **not** want to include normal clips.

Class mapping:

```text
fighting -> 0
smoking  -> 1
```

Expected folders:

```text
/home/mohammad/projects/mmaction2/dataset/v1/
  train/
    fighting/
    smoking/
  val/
    fighting/
    smoking/
  label_map.txt
  train.txt
  val.txt
```

The `normal/` folders may exist, but they are ignored as long as they are not listed in `train.txt` or `val.txt`.

## 1. Generate Binary Labels

Run this from the MMAction2 repo root:

```bash
cd /home/mohammad/projects/mmaction2
DATASET_ROOT=/home/mohammad/projects/mmaction2/dataset/v1

printf "fighting\nsmoking\n" > "$DATASET_ROOT/label_map.txt"

find "$DATASET_ROOT/train/fighting" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" \) \
  | sort | awk -v root="$DATASET_ROOT/train/" '{sub(root, "", $0); print $0 " 0"}' > "$DATASET_ROOT/train.txt"

find "$DATASET_ROOT/train/smoking" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" \) \
  | sort | awk -v root="$DATASET_ROOT/train/" '{sub(root, "", $0); print $0 " 1"}' >> "$DATASET_ROOT/train.txt"

find "$DATASET_ROOT/val/fighting" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" \) \
  | sort | awk -v root="$DATASET_ROOT/val/" '{sub(root, "", $0); print $0 " 0"}' > "$DATASET_ROOT/val.txt"

find "$DATASET_ROOT/val/smoking" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" \) \
  | sort | awk -v root="$DATASET_ROOT/val/" '{sub(root, "", $0); print $0 " 1"}' >> "$DATASET_ROOT/val.txt"
```

Check:

```bash
cat "$DATASET_ROOT/label_map.txt"
head "$DATASET_ROOT/train.txt"
head "$DATASET_ROOT/val.txt"
wc -l "$DATASET_ROOT/train.txt" "$DATASET_ROOT/val.txt"
```

For your current binary dataset, this should be:

```text
train.txt: 60
val.txt:   20
```

## 2. Binary Config

In `configs/recognition/tsn/tsn_r50_fighting_smoking.py`, use:

```python
_base_ = ['tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py']

dataset_type = 'VideoDataset'
data_root = '/home/mohammad/projects/mmaction2/dataset/v1/train'
data_root_val = '/home/mohammad/projects/mmaction2/dataset/v1/val'
ann_file_train = '/home/mohammad/projects/mmaction2/dataset/v1/train.txt'
ann_file_val = '/home/mohammad/projects/mmaction2/dataset/v1/val.txt'
ann_file_test = '/home/mohammad/projects/mmaction2/dataset/v1/val.txt'

model = dict(
    cls_head=dict(
        num_classes=2,
        average_clips='prob'))
```

The rest of the config can stay as already written: dataloaders, optimizer, schedule, evaluator, hooks, and `load_from`.

Most important line for binary training:

```python
num_classes=2
```

## 3. Verify Paths Before Training

Run:

```bash
DATASET_ROOT=/home/mohammad/projects/mmaction2/dataset/v1

while read video label; do
  test -f "$DATASET_ROOT/train/$video" || echo "Missing train file: $video"
done < "$DATASET_ROOT/train.txt"

while read video label; do
  test -f "$DATASET_ROOT/val/$video" || echo "Missing val file: $video"
done < "$DATASET_ROOT/val.txt"
```

If nothing prints, paths are good.

## 4. Browse a Few Samples

Run a quick sanity check:

```bash
python tools/visualizations/browse_dataset.py \
  configs/recognition/tsn/tsn_r50_fighting_smoking.py \
  browse_out_fighting_smoking \
  --mode pipeline \
  --show-number 6
```

Your earlier `FileNotFoundError` happened because `data_root` pointed to `/dataset/v1/train` while your real data is under `/home/mohammad/projects/mmaction2/dataset/v1/train`. The config now uses the real path.

## 5. Train Binary Model

```bash
python tools/train.py \
  configs/recognition/tsn/tsn_r50_fighting_smoking.py \
  --work-dir work_dirs/tsn_r50_fighting_smoking
```

If GPU memory is low, either lower `batch_size` in the config or use AMP:

```bash
python tools/train.py \
  configs/recognition/tsn/tsn_r50_fighting_smoking.py \
  --work-dir work_dirs/tsn_r50_fighting_smoking \
  --amp
```

Resume:

```bash
python tools/train.py \
  configs/recognition/tsn/tsn_r50_fighting_smoking.py \
  --work-dir work_dirs/tsn_r50_fighting_smoking \
  --resume work_dirs/tsn_r50_fighting_smoking/latest.pth
```

## 6. Test Binary Model

Replace `best_xxx.pth` with the real best checkpoint filename.

```bash
python tools/test.py \
  configs/recognition/tsn/tsn_r50_fighting_smoking.py \
  work_dirs/tsn_r50_fighting_smoking/best_xxx.pth
```

## 7. Inference

Single video:

```bash
python demo/demo.py \
  configs/recognition/tsn/tsn_r50_fighting_smoking.py \
  work_dirs/tsn_r50_fighting_smoking/best_xxx.pth \
  /path/to/new_video.mp4 \
  /home/mohammad/projects/mmaction2/dataset/v1/label_map.txt \
  --device cuda:0
```

Long video:

```bash
python demo/long_video_demo.py \
  configs/recognition/tsn/tsn_r50_fighting_smoking.py \
  work_dirs/tsn_r50_fighting_smoking/best_xxx.pth \
  /path/to/long_video.mp4 \
  /home/mohammad/projects/mmaction2/dataset/v1/label_map.txt \
  outputs/long_video_result.mp4 \
  --device cuda:0 \
  --threshold 0.5
```

## B. Optional Later: Add Normal Class

Use this setup later if you want the model to say `normal` when neither fighting nor smoking is happening.

Class mapping:

```text
normal   -> 0
fighting -> 1
smoking  -> 2
```

Expected folders:

```text
/home/mohammad/projects/mmaction2/dataset/v1/
  train/
    normal/
    fighting/
    smoking/
  val/
    normal/
    fighting/
    smoking/
```

Normal clips should be real negative examples: standing, walking, talking, sitting, using a phone, eating, drinking, touching face, crowded scenes, or any deployment-like footage where neither target action is present.

## 8. Generate 3-Class Labels

```bash
cd /home/mohammad/projects/mmaction2
DATASET_ROOT=/home/mohammad/projects/mmaction2/dataset/v1

printf "normal\nfighting\nsmoking\n" > "$DATASET_ROOT/label_map.txt"

find "$DATASET_ROOT/train/normal" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" \) \
  | sort | awk -v root="$DATASET_ROOT/train/" '{sub(root, "", $0); print $0 " 0"}' > "$DATASET_ROOT/train.txt"

find "$DATASET_ROOT/train/fighting" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" \) \
  | sort | awk -v root="$DATASET_ROOT/train/" '{sub(root, "", $0); print $0 " 1"}' >> "$DATASET_ROOT/train.txt"

find "$DATASET_ROOT/train/smoking" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" \) \
  | sort | awk -v root="$DATASET_ROOT/train/" '{sub(root, "", $0); print $0 " 2"}' >> "$DATASET_ROOT/train.txt"

find "$DATASET_ROOT/val/normal" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" \) \
  | sort | awk -v root="$DATASET_ROOT/val/" '{sub(root, "", $0); print $0 " 0"}' > "$DATASET_ROOT/val.txt"

find "$DATASET_ROOT/val/fighting" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" \) \
  | sort | awk -v root="$DATASET_ROOT/val/" '{sub(root, "", $0); print $0 " 1"}' >> "$DATASET_ROOT/val.txt"

find "$DATASET_ROOT/val/smoking" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" \) \
  | sort | awk -v root="$DATASET_ROOT/val/" '{sub(root, "", $0); print $0 " 2"}' >> "$DATASET_ROOT/val.txt"
```

Then change the config:

```python
model = dict(
    cls_head=dict(
        num_classes=3,
        average_clips='prob'))
```

After that, browse, train, test, and infer with the same commands as above.

## 9. Important Notes

- Binary model: will always choose `fighting` or `smoking`, even for a normal video.
- 3-class model: can learn `normal`, but only if you include enough real normal examples.
- After renaming or moving video files, always regenerate `train.txt` and `val.txt`.
- `ann_file_train` can be absolute, but paths inside `train.txt` are joined with `data_root`.
- Do not train if the path verification command prints missing files.
