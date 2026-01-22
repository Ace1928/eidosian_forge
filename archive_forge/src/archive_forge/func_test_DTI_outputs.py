from ..tensors import DTI
def test_DTI_outputs():
    output_map = dict(ad_file=dict(extensions=None), color_fa_file=dict(extensions=None), fa_file=dict(extensions=None), md_file=dict(extensions=None), out_file=dict(extensions=None), rd_file=dict(extensions=None))
    outputs = DTI.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value