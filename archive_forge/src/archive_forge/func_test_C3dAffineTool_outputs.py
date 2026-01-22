from ..c3 import C3dAffineTool
def test_C3dAffineTool_outputs():
    output_map = dict(itk_transform=dict(extensions=None))
    outputs = C3dAffineTool.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value