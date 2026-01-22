from ..model import MultipleRegressDesign
def test_MultipleRegressDesign_inputs():
    input_map = dict(contrasts=dict(mandatory=True), groups=dict(), regressors=dict(mandatory=True))
    inputs = MultipleRegressDesign.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value