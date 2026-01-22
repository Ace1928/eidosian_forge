from ..diffusion import DiffusionWeightedVolumeMasking
def test_DiffusionWeightedVolumeMasking_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), inputVolume=dict(argstr='%s', extensions=None, position=-4), otsuomegathreshold=dict(argstr='--otsuomegathreshold %f'), outputBaseline=dict(argstr='%s', hash_files=False, position=-2), removeislands=dict(argstr='--removeislands '), thresholdMask=dict(argstr='%s', hash_files=False, position=-1))
    inputs = DiffusionWeightedVolumeMasking.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value