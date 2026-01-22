from ..convert import ProcStreamlines
def test_ProcStreamlines_outputs():
    output_map = dict(outputroot_files=dict(), proc=dict(extensions=None))
    outputs = ProcStreamlines.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value