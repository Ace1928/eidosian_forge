from ..minc import Norm
def test_Norm_outputs():
    output_map = dict(output_file=dict(extensions=None), output_threshold_mask=dict(extensions=None))
    outputs = Norm.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value