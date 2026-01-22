from ..convert import DT2NIfTI
def test_DT2NIfTI_outputs():
    output_map = dict(dt=dict(extensions=None), exitcode=dict(extensions=None), lns0=dict(extensions=None))
    outputs = DT2NIfTI.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value