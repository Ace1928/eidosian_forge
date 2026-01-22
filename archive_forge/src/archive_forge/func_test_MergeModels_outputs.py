from ..surface import MergeModels
def test_MergeModels_outputs():
    output_map = dict(ModelOutput=dict(extensions=None, position=-1))
    outputs = MergeModels.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value