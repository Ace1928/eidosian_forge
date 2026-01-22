from ..epi import ApplyTOPUP
def test_ApplyTOPUP_outputs():
    output_map = dict(out_corrected=dict(extensions=None))
    outputs = ApplyTOPUP.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value