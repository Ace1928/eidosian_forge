from ..registration import MRICoreg
def test_MRICoreg_outputs():
    output_map = dict(out_lta_file=dict(extensions=None), out_params_file=dict(extensions=None), out_reg_file=dict(extensions=None))
    outputs = MRICoreg.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value