from ..preprocess import WatershedSkullStrip
def test_WatershedSkullStrip_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = WatershedSkullStrip.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value