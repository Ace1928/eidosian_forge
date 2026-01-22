from ..odf import MESD
def test_MESD_outputs():
    output_map = dict(mesd_data=dict(extensions=None))
    outputs = MESD.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value