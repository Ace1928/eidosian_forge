from ..featuredetection import DumpBinaryTrainingVectors
def test_DumpBinaryTrainingVectors_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), inputHeaderFilename=dict(argstr='--inputHeaderFilename %s', extensions=None), inputVectorFilename=dict(argstr='--inputVectorFilename %s', extensions=None))
    inputs = DumpBinaryTrainingVectors.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value