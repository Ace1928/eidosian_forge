from ..mesh import ComputeMeshWarp
def test_ComputeMeshWarp_outputs():
    output_map = dict(distance=dict(), out_file=dict(extensions=None), out_warp=dict(extensions=None))
    outputs = ComputeMeshWarp.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value