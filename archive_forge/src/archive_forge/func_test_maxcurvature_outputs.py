from ..maxcurvature import maxcurvature
def test_maxcurvature_outputs():
    output_map = dict(output=dict(extensions=None))
    outputs = maxcurvature.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value