from ..postproc import TrackMerge
def test_TrackMerge_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), output_file=dict(argstr='%s', extensions=None, position=-1, usedefault=True), track_files=dict(argstr='%s...', mandatory=True, position=0))
    inputs = TrackMerge.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value