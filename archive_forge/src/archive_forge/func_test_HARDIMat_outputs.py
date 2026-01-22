from ..odf import HARDIMat
def test_HARDIMat_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = HARDIMat.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value