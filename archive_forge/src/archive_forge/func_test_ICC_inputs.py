from ..icc import ICC
def test_ICC_inputs():
    input_map = dict(mask=dict(extensions=None, mandatory=True), subjects_sessions=dict(mandatory=True))
    inputs = ICC.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value