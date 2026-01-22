from ..filtering import OtsuThresholdImageFilter
def test_OtsuThresholdImageFilter_outputs():
    output_map = dict(outputVolume=dict(extensions=None, position=-1))
    outputs = OtsuThresholdImageFilter.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value