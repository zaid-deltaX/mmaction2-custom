# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
from pathlib import Path

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmaction.registry import RUNNERS

VIDEO_EXTENSIONS = {'.avi', '.mkv', '.mov', '.mp4', '.webm'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train an MMAction2 video recognition model on a '
        'folder-based custom dataset.')
    parser.add_argument('config', help='MMAction2 config file to train')
    parser.add_argument(
        'dataset_root',
        help='dataset root containing train/ and val/ class folders')
    parser.add_argument(
        '--work-dir',
        default=None,
        help='directory for logs and checkpoints')
    parser.add_argument(
        '--classes',
        nargs='+',
        default=None,
        help='class names in the exact label order. If omitted, class folders '
        'are sorted alphabetically.')
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=None,
        help='override train_cfg.max_epochs')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='override train/val dataloader batch_size')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='override train/val/test dataloader num_workers')
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=None,
        help='save checkpoint every N epochs')
    parser.add_argument(
        '--resume',
        nargs='?',
        const='auto',
        default=None,
        help='resume from latest checkpoint or a specified checkpoint')
    parser.add_argument(
        '--amp',
        action='store_true',
        help='enable automatic mixed precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override config options, e.g. model.cls_head.num_classes=2')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def discover_classes(dataset_root, classes):
    train_root = dataset_root / 'train'
    val_root = dataset_root / 'val'
    if not train_root.is_dir():
        raise FileNotFoundError(f'Missing train directory: {train_root}')
    if not val_root.is_dir():
        raise FileNotFoundError(f'Missing val directory: {val_root}')

    if classes:
        class_names = classes
    else:
        class_names = sorted(
            path.name for path in train_root.iterdir() if path.is_dir())

    if not class_names:
        raise RuntimeError(f'No class folders found in {train_root}')

    for split_root in (train_root, val_root):
        missing = [name for name in class_names if not (split_root / name).is_dir()]
        if missing:
            raise FileNotFoundError(
                f'Missing class folders in {split_root}: {missing}')

    return class_names


def iter_video_files(class_dir):
    return sorted(
        path for path in class_dir.rglob('*')
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)


def write_annotation_file(dataset_root, split, class_names):
    split_root = dataset_root / split
    ann_file = dataset_root / f'{split}.txt'
    lines = []

    for label, class_name in enumerate(class_names):
        class_dir = split_root / class_name
        for video_path in iter_video_files(class_dir):
            rel_path = video_path.relative_to(split_root).as_posix()
            lines.append(f'{rel_path} {label}')

    if not lines:
        raise RuntimeError(f'No videos found under {split_root}')

    ann_file.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return ann_file, len(lines)


def prepare_dataset_files(dataset_root, classes):
    dataset_root = Path(dataset_root).resolve()
    class_names = discover_classes(dataset_root, classes)

    label_map = dataset_root / 'label_map.txt'
    label_map.write_text('\n'.join(class_names) + '\n', encoding='utf-8')

    train_ann, train_count = write_annotation_file(dataset_root, 'train',
                                                  class_names)
    val_ann, val_count = write_annotation_file(dataset_root, 'val',
                                              class_names)

    print('Prepared dataset files')
    print(f'  Dataset root: {dataset_root}')
    print(f'  Classes: {class_names}')
    print(f'  Label map: {label_map}')
    print(f'  Train annotations: {train_ann} ({train_count} videos)')
    print(f'  Val annotations: {val_ann} ({val_count} videos)')

    return dataset_root, class_names, train_ann, val_ann


def set_num_classes(model_cfg, num_classes):
    if 'cls_head' in model_cfg:
        model_cfg.cls_head.num_classes = num_classes
    if 'data_preprocessor' in model_cfg:
        blending = model_cfg.data_preprocessor.get('blending', None)
        if blending and 'augments' in blending:
            for augment in blending.augments:
                if 'num_classes' in augment:
                    augment.num_classes = num_classes


def patch_dataloader(dataloader, ann_file, data_root, batch_size, num_workers):
    dataloader.dataset.ann_file = str(ann_file)
    dataloader.dataset.data_prefix = dict(video=str(data_root))
    if batch_size is not None:
        dataloader.batch_size = batch_size
    if num_workers is not None:
        dataloader.num_workers = num_workers
        dataloader.persistent_workers = num_workers > 0


def patch_config(cfg, args, dataset_root, class_names, train_ann, val_ann):
    train_root = dataset_root / 'train'
    val_root = dataset_root / 'val'

    set_num_classes(cfg.model, len(class_names))
    patch_dataloader(cfg.train_dataloader, train_ann, train_root,
                     args.batch_size, args.num_workers)
    patch_dataloader(cfg.val_dataloader, val_ann, val_root, args.batch_size,
                     args.num_workers)
    patch_dataloader(cfg.test_dataloader, val_ann, val_root, 1,
                     args.num_workers)

    if args.max_epochs is not None:
        cfg.train_cfg.max_epochs = args.max_epochs
    if args.checkpoint_interval is not None:
        cfg.default_hooks.checkpoint.interval = args.checkpoint_interval

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        config_name = Path(args.config).stem
        cfg.work_dir = str(Path('work_dirs') / config_name)

    if args.resume is not None:
        cfg.resume = True
        if args.resume != 'auto':
            cfg.load_from = args.resume

    if args.amp:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    return cfg


def main():
    args = parse_args()
    dataset_root, class_names, train_ann, val_ann = prepare_dataset_files(
        args.dataset_root, args.classes)

    cfg = Config.fromfile(args.config)
    cfg.launcher = 'none'
    cfg = patch_config(cfg, args, dataset_root, class_names, train_ann,
                       val_ann)

    print('\nTraining setup')
    print(f'  Config: {args.config}')
    print(f'  Work dir: {cfg.work_dir}')
    print(f'  Classes: {len(class_names)}')
    print(f'  Max epochs: {cfg.train_cfg.max_epochs}')

    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)
    runner.train()


if __name__ == '__main__':
    main()
