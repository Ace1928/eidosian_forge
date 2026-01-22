from ..diffusion import DiffusionWeightedVolumeMasking
def test_DiffusionWeightedVolumeMasking_outputs():
    output_map = dict(outputBaseline=dict(extensions=None, position=-2), thresholdMask=dict(extensions=None, position=-1))
    outputs = DiffusionWeightedVolumeMasking.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value