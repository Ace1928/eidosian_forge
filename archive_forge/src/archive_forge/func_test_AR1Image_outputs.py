from ..maths import AR1Image
def test_AR1Image_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = AR1Image.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value