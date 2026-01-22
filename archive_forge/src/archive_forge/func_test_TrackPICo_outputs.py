from ..dti import TrackPICo
def test_TrackPICo_outputs():
    output_map = dict(tracked=dict(extensions=None))
    outputs = TrackPICo.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value