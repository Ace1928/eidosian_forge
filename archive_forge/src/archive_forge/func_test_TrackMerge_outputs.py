from ..postproc import TrackMerge
def test_TrackMerge_outputs():
    output_map = dict(track_file=dict(extensions=None))
    outputs = TrackMerge.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value