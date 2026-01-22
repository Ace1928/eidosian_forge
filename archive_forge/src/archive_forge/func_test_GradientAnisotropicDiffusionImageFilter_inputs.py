from ..featuredetection import GradientAnisotropicDiffusionImageFilter
def test_GradientAnisotropicDiffusionImageFilter_inputs():
    input_map = dict(args=dict(argstr='%s'), conductance=dict(argstr='--conductance %f'), environ=dict(nohash=True, usedefault=True), inputVolume=dict(argstr='--inputVolume %s', extensions=None), numberOfIterations=dict(argstr='--numberOfIterations %d'), outputVolume=dict(argstr='--outputVolume %s', hash_files=False), timeStep=dict(argstr='--timeStep %f'))
    inputs = GradientAnisotropicDiffusionImageFilter.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value