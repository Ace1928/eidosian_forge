from ..diffusion import DWIToDTIEstimation
def test_DWIToDTIEstimation_outputs():
    output_map = dict(outputBaseline=dict(extensions=None, position=-1), outputTensor=dict(extensions=None, position=-2))
    outputs = DWIToDTIEstimation.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value