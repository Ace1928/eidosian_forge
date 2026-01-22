from ..gtract import gtractCopyImageOrientation
def test_gtractCopyImageOrientation_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), inputReferenceVolume=dict(argstr='--inputReferenceVolume %s', extensions=None), inputVolume=dict(argstr='--inputVolume %s', extensions=None), numberOfThreads=dict(argstr='--numberOfThreads %d'), outputVolume=dict(argstr='--outputVolume %s', hash_files=False))
    inputs = gtractCopyImageOrientation.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value