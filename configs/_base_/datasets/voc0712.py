# dataset settings
dataset_type = 'VOCDataset'
data_root = '/home/kai/Desktop/VOC_format'
# img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_norm_cfg = dict(mean=[110.364, 111.8685, 107.1765], std=[52.173, 51.6375, 55.386], to_rgb=True)
# tensor([0.4328, 0.4387, 0.4203])
# tensor([0.2046, 0.2025, 0.2172])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(224, 224), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(224, 224),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=32,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                '/home/kai/Desktop/VOC_format/ImageSet/Main/train.txt',
                '/home/kai/Desktop/VOC_format/ImageSet/Main/train.txt'
            ],
            img_prefix=['/home/kai/Desktop/VOC_format', '/home/kai/Desktop/VOC_format'],
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file='/home/kai/Desktop/VOC_format/ImageSet/Main/test.txt',
        img_prefix='/home/kai/Desktop/VOC_format',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='/home/kai/Desktop/VOC_format/ImageSet/Main/test.txt',
        img_prefix='/home/kai/Desktop/VOC_format',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mAP')
