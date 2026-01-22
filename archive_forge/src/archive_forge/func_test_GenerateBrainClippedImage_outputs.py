from ..featuredetection import GenerateBrainClippedImage
def test_GenerateBrainClippedImage_outputs():
    output_map = dict(outputFileName=dict(extensions=None))
    outputs = GenerateBrainClippedImage.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value