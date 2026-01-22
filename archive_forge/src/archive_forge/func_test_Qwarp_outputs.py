from ..preprocess import Qwarp
def test_Qwarp_outputs():
    output_map = dict(base_warp=dict(extensions=None), source_warp=dict(extensions=None), warped_base=dict(extensions=None), warped_source=dict(extensions=None), weights=dict(extensions=None))
    outputs = Qwarp.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value