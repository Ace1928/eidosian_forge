from ..morphology import GrayscaleFillHoleImageFilter
def test_GrayscaleFillHoleImageFilter_outputs():
    output_map = dict(outputVolume=dict(extensions=None, position=-1))
    outputs = GrayscaleFillHoleImageFilter.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value