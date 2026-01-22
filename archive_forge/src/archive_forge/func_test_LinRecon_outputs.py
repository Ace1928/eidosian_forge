from ..odf import LinRecon
def test_LinRecon_outputs():
    output_map = dict(recon_data=dict(extensions=None))
    outputs = LinRecon.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value