from ..diffusion import DTIexport
def test_DTIexport_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), inputTensor=dict(argstr='%s', extensions=None, position=-2), outputFile=dict(argstr='%s', hash_files=False, position=-1))
    inputs = DTIexport.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value