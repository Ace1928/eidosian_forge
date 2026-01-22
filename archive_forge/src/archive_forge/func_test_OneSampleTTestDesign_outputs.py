from ..model import OneSampleTTestDesign
def test_OneSampleTTestDesign_outputs():
    output_map = dict(spm_mat_file=dict(extensions=None))
    outputs = OneSampleTTestDesign.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value