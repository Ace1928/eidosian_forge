from ..brains import BRAINSMush
def test_BRAINSMush_outputs():
    output_map = dict(outputMask=dict(extensions=None), outputVolume=dict(extensions=None), outputWeightsFile=dict(extensions=None))
    outputs = BRAINSMush.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value