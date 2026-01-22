from ..preprocess import Warp
def test_Warp_outputs():
    output_map = dict(out_file=dict(extensions=None), warp_file=dict(extensions=None))
    outputs = Warp.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value