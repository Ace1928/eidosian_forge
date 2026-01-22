from ..denoising import MedianImageFilter
def test_MedianImageFilter_outputs():
    output_map = dict(outputVolume=dict(extensions=None, position=-1))
    outputs = MedianImageFilter.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value