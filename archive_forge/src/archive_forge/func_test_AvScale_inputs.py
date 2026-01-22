from ..utils import AvScale
def test_AvScale_inputs():
    input_map = dict(all_param=dict(argstr='--allparams'), args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), mat_file=dict(argstr='%s', extensions=None, position=-2), ref_file=dict(argstr='%s', extensions=None, position=-1))
    inputs = AvScale.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value