from ..registration import PointsWarp
def test_PointsWarp_outputs():
    output_map = dict(warped_file=dict(extensions=None))
    outputs = PointsWarp.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value