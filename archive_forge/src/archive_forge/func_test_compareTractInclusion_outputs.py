from ..gtract import compareTractInclusion
def test_compareTractInclusion_outputs():
    output_map = dict()
    outputs = compareTractInclusion.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value