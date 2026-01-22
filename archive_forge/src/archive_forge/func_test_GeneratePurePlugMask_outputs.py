from ..utilities import GeneratePurePlugMask
def test_GeneratePurePlugMask_outputs():
    output_map = dict(outputMaskFile=dict(extensions=None))
    outputs = GeneratePurePlugMask.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value