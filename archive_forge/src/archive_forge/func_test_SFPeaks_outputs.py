from ..odf import SFPeaks
def test_SFPeaks_outputs():
    output_map = dict(peaks=dict(extensions=None))
    outputs = SFPeaks.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value