from ..utils import MRConvert
def test_MRConvert_outputs():
    output_map = dict(json_export=dict(extensions=None), out_bval=dict(extensions=None), out_bvec=dict(extensions=None), out_file=dict(extensions=None))
    outputs = MRConvert.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value