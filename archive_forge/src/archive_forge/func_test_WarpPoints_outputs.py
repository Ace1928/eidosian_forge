from ..mesh import WarpPoints
def test_WarpPoints_outputs():
    output_map = dict(out_points=dict(extensions=None))
    outputs = WarpPoints.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value