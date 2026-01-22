from ..utils import WarpPointsToStd
def test_WarpPointsToStd_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = WarpPointsToStd.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value