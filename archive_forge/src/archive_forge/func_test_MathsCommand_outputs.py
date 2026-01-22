from ..maths import MathsCommand
def test_MathsCommand_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = MathsCommand.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value