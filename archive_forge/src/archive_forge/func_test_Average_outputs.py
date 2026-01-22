from ..minc import Average
def test_Average_outputs():
    output_map = dict(output_file=dict(extensions=None))
    outputs = Average.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value