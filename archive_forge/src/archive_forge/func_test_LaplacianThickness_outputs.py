from ..segmentation import LaplacianThickness
def test_LaplacianThickness_outputs():
    output_map = dict(output_image=dict(extensions=None))
    outputs = LaplacianThickness.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value