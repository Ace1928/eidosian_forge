from ..fix import FeatureExtractor
def test_FeatureExtractor_outputs():
    output_map = dict(mel_ica=dict(argstr='%s', copyfile=False, position=-1))
    outputs = FeatureExtractor.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value