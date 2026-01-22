from ..dti import TrackBedpostxDeter
def test_TrackBedpostxDeter_outputs():
    output_map = dict(tracked=dict(extensions=None))
    outputs = TrackBedpostxDeter.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value