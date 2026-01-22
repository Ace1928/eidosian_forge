from ..brainsuite import Pialmesh
def test_Pialmesh_outputs():
    output_map = dict(outputSurfaceFile=dict(extensions=None))
    outputs = Pialmesh.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value