from ..epi import EddyCorrect
def test_EddyCorrect_outputs():
    output_map = dict(eddy_corrected=dict(extensions=None))
    outputs = EddyCorrect.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value