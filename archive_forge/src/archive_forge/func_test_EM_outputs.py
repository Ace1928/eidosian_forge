from ..em import EM
def test_EM_outputs():
    output_map = dict(out_bc_file=dict(extensions=None), out_file=dict(extensions=None), out_outlier_file=dict(extensions=None))
    outputs = EM.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value