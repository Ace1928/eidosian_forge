from ..model import FEAT
def test_FEAT_outputs():
    output_map = dict(feat_dir=dict())
    outputs = FEAT.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value