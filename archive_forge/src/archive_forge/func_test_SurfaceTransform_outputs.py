from ..utils import SurfaceTransform
def test_SurfaceTransform_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = SurfaceTransform.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value