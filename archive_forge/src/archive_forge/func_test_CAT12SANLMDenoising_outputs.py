from ..preprocess import CAT12SANLMDenoising
def test_CAT12SANLMDenoising_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = CAT12SANLMDenoising.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value