from ..preprocess import SpaceTimeRealigner
def test_SpaceTimeRealigner_outputs():
    output_map = dict(out_file=dict(), par_file=dict())
    outputs = SpaceTimeRealigner.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value