_base_ = ['tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py']

dataset_type = 'VideoDataset'
data_root = r'C:\Projects\action_recognition\mmaction2\dataset\v1\train'
data_root_val = r'C:\Projects\action_recognition\mmaction2\dataset\v1\val'
ann_file_train = r'C:\Projects\action_recognition\mmaction2\dataset\v1\train.txt'
ann_file_val = r'C:\Projects\action_recognition\mmaction2\dataset\v1\val.txt'
ann_file_test = r'C:\Projects\action_recognition\mmaction2\dataset\v1\val.txt'

model = dict(
    cls_head=dict(
        num_classes=2,
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

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,
    val_begin=1,
    val_interval=1)

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
    optimizer=dict(
        type='SGD',
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=3, save_best='auto'))

load_from = 'https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth'
