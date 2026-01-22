from ..utils import TCK2VTK
def test_TCK2VTK_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = TCK2VTK.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value