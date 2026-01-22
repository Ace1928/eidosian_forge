from ..preprocess import GenerateWhiteMatterMask
def test_GenerateWhiteMatterMask_outputs():
    output_map = dict(WMprobabilitymap=dict(extensions=None))
    outputs = GenerateWhiteMatterMask.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value