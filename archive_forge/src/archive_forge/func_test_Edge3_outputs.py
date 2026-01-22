from ..utils import Edge3
def test_Edge3_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = Edge3.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value