from ..model import SegStats
def test_SegStats_outputs():
    output_map = dict(avgwf_file=dict(extensions=None), avgwf_txt_file=dict(extensions=None), sf_avg_file=dict(extensions=None), summary_file=dict(extensions=None))
    outputs = SegStats.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value