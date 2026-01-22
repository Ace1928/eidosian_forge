from ..brainsuite import Hemisplit
def test_Hemisplit_outputs():
    output_map = dict(outputLeftHemisphere=dict(extensions=None), outputLeftPialHemisphere=dict(extensions=None), outputRightHemisphere=dict(extensions=None), outputRightPialHemisphere=dict(extensions=None))
    outputs = Hemisplit.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value