from ..minc import XfmAvg
def test_XfmAvg_outputs():
    output_map = dict(output_file=dict(extensions=None), output_grid=dict(extensions=None))
    outputs = XfmAvg.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value