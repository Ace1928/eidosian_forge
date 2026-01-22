from ..brains import landmarksConstellationWeights
def test_landmarksConstellationWeights_inputs():
    input_map = dict(LLSModel=dict(argstr='--LLSModel %s', extensions=None), args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), inputTemplateModel=dict(argstr='--inputTemplateModel %s', extensions=None), inputTrainingList=dict(argstr='--inputTrainingList %s', extensions=None), outputWeightsList=dict(argstr='--outputWeightsList %s', hash_files=False))
    inputs = landmarksConstellationWeights.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value