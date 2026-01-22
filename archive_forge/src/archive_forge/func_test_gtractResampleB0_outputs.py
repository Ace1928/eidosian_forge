from ..gtract import gtractResampleB0
def test_gtractResampleB0_outputs():
    output_map = dict(outputVolume=dict(extensions=None))
    outputs = gtractResampleB0.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value