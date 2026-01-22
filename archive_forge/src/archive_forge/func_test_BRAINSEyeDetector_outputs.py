from ..brains import BRAINSEyeDetector
def test_BRAINSEyeDetector_outputs():
    output_map = dict(outputVolume=dict(extensions=None))
    outputs = BRAINSEyeDetector.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value