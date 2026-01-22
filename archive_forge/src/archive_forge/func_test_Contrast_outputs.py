from ..utils import Contrast
def test_Contrast_outputs():
    output_map = dict(out_contrast=dict(extensions=None), out_log=dict(extensions=None), out_stats=dict(extensions=None))
    outputs = Contrast.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value