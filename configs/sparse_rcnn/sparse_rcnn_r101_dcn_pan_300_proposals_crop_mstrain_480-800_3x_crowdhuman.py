_base_ = './sparse_rcnn_r50_dcn_fpn_300_proposals_crop_mstrain_480-800_1x_crowdhuman.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))

checkpoint_config = dict(interval=3)
lr_config = dict(policy='step', step=[24,30])
runner = dict(type='EpochBasedRunner', max_epochs=36)

# /root/.cache/torch/hub/checkpoints/
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20201223_023452-c23c3564.pth'