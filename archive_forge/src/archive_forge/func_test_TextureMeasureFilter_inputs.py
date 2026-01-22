from ..featuredetection import TextureMeasureFilter
def test_TextureMeasureFilter_inputs():
    input_map = dict(args=dict(argstr='%s'), distance=dict(argstr='--distance %d'), environ=dict(nohash=True, usedefault=True), inputMaskVolume=dict(argstr='--inputMaskVolume %s', extensions=None), inputVolume=dict(argstr='--inputVolume %s', extensions=None), insideROIValue=dict(argstr='--insideROIValue %f'), outputFilename=dict(argstr='--outputFilename %s', hash_files=False))
    inputs = TextureMeasureFilter.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value