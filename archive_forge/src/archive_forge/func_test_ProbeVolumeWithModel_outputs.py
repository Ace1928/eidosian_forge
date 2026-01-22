from ..surface import ProbeVolumeWithModel
def test_ProbeVolumeWithModel_outputs():
    output_map = dict(OutputModel=dict(extensions=None, position=-1))
    outputs = ProbeVolumeWithModel.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value