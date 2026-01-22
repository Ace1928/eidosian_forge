from ..preprocess import MRMultiply
def test_MRMultiply_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = MRMultiply.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value