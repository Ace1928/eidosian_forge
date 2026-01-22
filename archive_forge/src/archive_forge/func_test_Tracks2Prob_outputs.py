from ..tracking import Tracks2Prob
def test_Tracks2Prob_outputs():
    output_map = dict(tract_image=dict(extensions=None))
    outputs = Tracks2Prob.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value