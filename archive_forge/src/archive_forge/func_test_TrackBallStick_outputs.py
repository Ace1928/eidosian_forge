from ..dti import TrackBallStick
def test_TrackBallStick_outputs():
    output_map = dict(tracked=dict(extensions=None))
    outputs = TrackBallStick.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value