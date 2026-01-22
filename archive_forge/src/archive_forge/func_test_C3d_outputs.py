from ..c3 import C3d
def test_C3d_outputs():
    output_map = dict(out_files=dict())
    outputs = C3d.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value