from ..utils import NwarpAdjust
def test_NwarpAdjust_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), in_files=dict(argstr='-source %s'), num_threads=dict(nohash=True, usedefault=True), out_file=dict(argstr='-prefix %s', extensions=None, keep_extension=True, name_source='in_files', name_template='%s_NwarpAdjust', requires=['in_files']), outputtype=dict(), warps=dict(argstr='-nwarp %s', mandatory=True))
    inputs = NwarpAdjust.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value