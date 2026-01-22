from ..preprocess import RealignUnwarp
def test_RealignUnwarp_outputs():
    output_map = dict(mean_image=dict(extensions=None), modified_in_files=dict(), realigned_unwarped_files=dict(), realignment_parameters=dict())
    outputs = RealignUnwarp.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value