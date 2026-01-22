from ..segmentation import BRAINSTalairach
def test_BRAINSTalairach_outputs():
    output_map = dict(outputBox=dict(extensions=None), outputGrid=dict(extensions=None))
    outputs = BRAINSTalairach.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value