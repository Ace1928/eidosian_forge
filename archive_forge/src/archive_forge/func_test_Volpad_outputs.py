from ..minc import Volpad
def test_Volpad_outputs():
    output_map = dict(output_file=dict(extensions=None))
    outputs = Volpad.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value