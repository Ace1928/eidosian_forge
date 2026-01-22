from ..utils import SH2Amp
def test_SH2Amp_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = SH2Amp.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value