from ..gtract import gtractCoregBvalues
def test_gtractCoregBvalues_outputs():
    output_map = dict(outputTransform=dict(extensions=None), outputVolume=dict(extensions=None))
    outputs = gtractCoregBvalues.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value