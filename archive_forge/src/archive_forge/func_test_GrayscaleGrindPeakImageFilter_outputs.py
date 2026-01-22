from ..morphology import GrayscaleGrindPeakImageFilter
def test_GrayscaleGrindPeakImageFilter_outputs():
    output_map = dict(outputVolume=dict(extensions=None, position=-1))
    outputs = GrayscaleGrindPeakImageFilter.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value