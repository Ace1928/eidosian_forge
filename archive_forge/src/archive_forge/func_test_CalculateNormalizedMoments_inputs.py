from ..misc import CalculateNormalizedMoments
def test_CalculateNormalizedMoments_inputs():
    input_map = dict(moment=dict(mandatory=True), timeseries_file=dict(extensions=None, mandatory=True))
    inputs = CalculateNormalizedMoments.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value