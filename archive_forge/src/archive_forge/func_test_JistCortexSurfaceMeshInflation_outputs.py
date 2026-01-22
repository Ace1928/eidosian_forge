from ..developer import JistCortexSurfaceMeshInflation
def test_JistCortexSurfaceMeshInflation_outputs():
    output_map = dict(outInflated=dict(extensions=None), outOriginal=dict(extensions=None))
    outputs = JistCortexSurfaceMeshInflation.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value