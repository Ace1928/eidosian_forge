from ..utilities import HistogramMatchingFilter
def test_HistogramMatchingFilter_outputs():
    output_map = dict(outputVolume=dict(extensions=None))
    outputs = HistogramMatchingFilter.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value