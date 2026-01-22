from ..minc import BestLinReg
def test_BestLinReg_outputs():
    output_map = dict(output_mnc=dict(extensions=None), output_xfm=dict(extensions=None))
    outputs = BestLinReg.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value