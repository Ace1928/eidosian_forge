from ..preprocess import ApplyXFM
def test_ApplyXFM_outputs():
    output_map = dict(out_file=dict(extensions=None), out_log=dict(extensions=None), out_matrix_file=dict(extensions=None))
    outputs = ApplyXFM.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value