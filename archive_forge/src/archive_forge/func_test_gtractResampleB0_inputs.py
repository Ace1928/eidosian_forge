from ..gtract import gtractResampleB0
def test_gtractResampleB0_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), inputAnatomicalVolume=dict(argstr='--inputAnatomicalVolume %s', extensions=None), inputTransform=dict(argstr='--inputTransform %s', extensions=None), inputVolume=dict(argstr='--inputVolume %s', extensions=None), numberOfThreads=dict(argstr='--numberOfThreads %d'), outputVolume=dict(argstr='--outputVolume %s', hash_files=False), transformType=dict(argstr='--transformType %s'), vectorIndex=dict(argstr='--vectorIndex %d'))
    inputs = gtractResampleB0.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value