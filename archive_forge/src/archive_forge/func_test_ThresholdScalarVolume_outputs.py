from ..thresholdscalarvolume import ThresholdScalarVolume
def test_ThresholdScalarVolume_outputs():
    output_map = dict(OutputVolume=dict(extensions=None, position=-1))
    outputs = ThresholdScalarVolume.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value