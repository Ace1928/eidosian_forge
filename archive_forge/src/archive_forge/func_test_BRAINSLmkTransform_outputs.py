from ..brains import BRAINSLmkTransform
def test_BRAINSLmkTransform_outputs():
    output_map = dict(outputAffineTransform=dict(extensions=None), outputResampledVolume=dict(extensions=None))
    outputs = BRAINSLmkTransform.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value