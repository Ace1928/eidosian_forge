from ..brains import landmarksConstellationAligner
def test_landmarksConstellationAligner_outputs():
    output_map = dict(outputLandmarksPaired=dict(extensions=None))
    outputs = landmarksConstellationAligner.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value