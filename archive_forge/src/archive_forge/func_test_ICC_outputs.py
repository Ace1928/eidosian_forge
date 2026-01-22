from ..icc import ICC
def test_ICC_outputs():
    output_map = dict(icc_map=dict(extensions=None), session_var_map=dict(extensions=None), subject_var_map=dict(extensions=None))
    outputs = ICC.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value