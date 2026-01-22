from ..base import Split
def test_Split_outputs():
    output_map = dict()
    outputs = Split.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value