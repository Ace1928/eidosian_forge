from ..utils import Unifize
def test_Unifize_outputs():
    output_map = dict(out_file=dict(extensions=None), scale_file=dict(extensions=None))
    outputs = Unifize.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value