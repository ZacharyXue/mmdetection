_base_ = './sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_1x_crowdhuman.py'

model = dict(
    backbone=dict(
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    # neck=dict(
    #     type='PAFPN',
    #     in_channels=[256, 512, 1024, 2048],
    #     out_channels=256,
    #     start_level=0,
    #     add_extra_convs='on_input',
    #     num_outs=4
    # )
)

runner = dict(type='EpochBasedRunner', max_epochs=12)

load_from = './checkpoints/sparse_rcnn_r50_fpn_2x_300p/epoch_24.pth'