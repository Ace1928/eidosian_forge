from ..model import TwoSampleTTestDesign
def test_TwoSampleTTestDesign_outputs():
    output_map = dict(spm_mat_file=dict(extensions=None))
    outputs = TwoSampleTTestDesign.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value