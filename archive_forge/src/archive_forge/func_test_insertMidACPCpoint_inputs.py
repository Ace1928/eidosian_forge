from ..brains import insertMidACPCpoint
def test_insertMidACPCpoint_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), inputLandmarkFile=dict(argstr='--inputLandmarkFile %s', extensions=None), outputLandmarkFile=dict(argstr='--outputLandmarkFile %s', hash_files=False))
    inputs = insertMidACPCpoint.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value