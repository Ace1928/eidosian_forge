from ..utils import InvWarp
def test_InvWarp_outputs():
    output_map = dict(inverse_warp=dict(extensions=None))
    outputs = InvWarp.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value