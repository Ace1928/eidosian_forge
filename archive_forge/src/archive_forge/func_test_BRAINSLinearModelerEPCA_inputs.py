from ..brains import BRAINSLinearModelerEPCA
def test_BRAINSLinearModelerEPCA_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), inputTrainingList=dict(argstr='--inputTrainingList %s', extensions=None), numberOfThreads=dict(argstr='--numberOfThreads %d'))
    inputs = BRAINSLinearModelerEPCA.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value