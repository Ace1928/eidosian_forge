from ..utils import ParcellationStats
def test_ParcellationStats_outputs():
    output_map = dict(out_color=dict(extensions=None), out_table=dict(extensions=None))
    outputs = ParcellationStats.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value