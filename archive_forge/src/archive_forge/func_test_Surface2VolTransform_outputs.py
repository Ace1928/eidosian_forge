from ..utils import Surface2VolTransform
def test_Surface2VolTransform_outputs():
    output_map = dict(transformed_file=dict(extensions=None), vertexvol_file=dict(extensions=None))
    outputs = Surface2VolTransform.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value