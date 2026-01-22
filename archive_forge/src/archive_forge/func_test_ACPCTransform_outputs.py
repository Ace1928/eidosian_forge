from ..specialized import ACPCTransform
def test_ACPCTransform_outputs():
    output_map = dict(outputTransform=dict(extensions=None))
    outputs = ACPCTransform.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value