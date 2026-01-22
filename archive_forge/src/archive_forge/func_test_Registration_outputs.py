from ..registration import Registration
def test_Registration_outputs():
    output_map = dict(composite_transform=dict(extensions=None), elapsed_time=dict(), forward_invert_flags=dict(), forward_transforms=dict(), inverse_composite_transform=dict(extensions=None), inverse_warped_image=dict(extensions=None), metric_value=dict(), reverse_forward_invert_flags=dict(), reverse_forward_transforms=dict(), reverse_invert_flags=dict(), reverse_transforms=dict(), save_state=dict(extensions=None), warped_image=dict(extensions=None))
    outputs = Registration.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value