from ..gtract import gtractCostFastMarching
def test_gtractCostFastMarching_outputs():
    output_map = dict(outputCostVolume=dict(extensions=None), outputSpeedVolume=dict(extensions=None))
    outputs = gtractCostFastMarching.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value