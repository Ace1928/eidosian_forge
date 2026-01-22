from ..stats import ActivationCount
def test_ActivationCount_outputs():
    output_map = dict(acm_neg=dict(extensions=None), acm_pos=dict(extensions=None), out_file=dict(extensions=None))
    outputs = ActivationCount.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value