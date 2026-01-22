from ..commandlineonly import fiberstats
def test_fiberstats_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), fiber_file=dict(argstr='--fiber_file %s', extensions=None), verbose=dict(argstr='--verbose '))
    inputs = fiberstats.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value