from ..preprocess import ReplaceFSwithFIRST
def test_ReplaceFSwithFIRST_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), in_config=dict(argstr='%s', extensions=None, position=-2), in_file=dict(argstr='%s', extensions=None, mandatory=True, position=-4), in_t1w=dict(argstr='%s', extensions=None, mandatory=True, position=-3), out_file=dict(argstr='%s', extensions=None, mandatory=True, position=-1, usedefault=True))
    inputs = ReplaceFSwithFIRST.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value