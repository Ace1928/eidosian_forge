from ..registration import CompositeTransformUtil
def test_CompositeTransformUtil_outputs():
    output_map = dict(affine_transform=dict(extensions=None), displacement_field=dict(extensions=None), out_file=dict(extensions=None))
    outputs = CompositeTransformUtil.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value