from ..arithmetic import CastScalarVolume
def test_CastScalarVolume_outputs():
    output_map = dict(OutputVolume=dict(extensions=None, position=-1))
    outputs = CastScalarVolume.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value