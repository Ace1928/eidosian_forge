from ..filtering import ResampleScalarVolume
def test_ResampleScalarVolume_outputs():
    output_map = dict(OutputVolume=dict(extensions=None, position=-1))
    outputs = ResampleScalarVolume.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value