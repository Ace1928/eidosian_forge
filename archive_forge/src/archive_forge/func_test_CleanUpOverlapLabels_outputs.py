from ..brains import CleanUpOverlapLabels
def test_CleanUpOverlapLabels_outputs():
    output_map = dict(outputBinaryVolumes=dict())
    outputs = CleanUpOverlapLabels.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value