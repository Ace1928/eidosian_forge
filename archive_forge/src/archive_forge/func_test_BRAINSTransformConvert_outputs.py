from ..brains import BRAINSTransformConvert
def test_BRAINSTransformConvert_outputs():
    output_map = dict(displacementVolume=dict(extensions=None), outputTransform=dict(extensions=None))
    outputs = BRAINSTransformConvert.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value