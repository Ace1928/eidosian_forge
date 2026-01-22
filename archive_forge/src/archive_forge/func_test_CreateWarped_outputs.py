from ..preprocess import CreateWarped
def test_CreateWarped_outputs():
    output_map = dict(warped_files=dict())
    outputs = CreateWarped.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value