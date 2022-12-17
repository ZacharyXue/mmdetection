_base_ = './sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py'

model = dict(
    backbone=dict(
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    neck=dict(
        type='PAFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=4
    )
)

# runner = dict(type='EpochBasedRunner', max_epochs=50)
runner = dict(type='EpochBasedRunner', max_epochs=36)
# runner = dict(type='EpochBasedRunner', max_epochs=18)
lr_config = dict(policy='step', step=[27, 36])

checkpoint_config = dict(interval=6)

# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20201223_024605-9fe92701.pth'