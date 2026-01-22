from ..denoising import DWIUnbiasedNonLocalMeansFilter
def test_DWIUnbiasedNonLocalMeansFilter_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), hp=dict(argstr='--hp %f'), inputVolume=dict(argstr='%s', extensions=None, position=-2), ng=dict(argstr='--ng %d'), outputVolume=dict(argstr='%s', hash_files=False, position=-1), rc=dict(argstr='--rc %s', sep=','), re=dict(argstr='--re %s', sep=','), rs=dict(argstr='--rs %s', sep=','))
    inputs = DWIUnbiasedNonLocalMeansFilter.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value