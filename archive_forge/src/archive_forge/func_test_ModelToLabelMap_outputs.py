from ..surface import ModelToLabelMap
def test_ModelToLabelMap_outputs():
    output_map = dict(OutputVolume=dict(extensions=None, position=-1))
    outputs = ModelToLabelMap.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value