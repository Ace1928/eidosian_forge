from ..epi import PrepareFieldmap
def test_PrepareFieldmap_outputs():
    output_map = dict(out_fieldmap=dict(extensions=None))
    outputs = PrepareFieldmap.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value