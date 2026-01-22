from ..minc import Convert
def test_Convert_outputs():
    output_map = dict(output_file=dict(extensions=None))
    outputs = Convert.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value