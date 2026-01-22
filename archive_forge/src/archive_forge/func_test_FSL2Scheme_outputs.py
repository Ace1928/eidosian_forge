from ..convert import FSL2Scheme
def test_FSL2Scheme_outputs():
    output_map = dict(scheme=dict(extensions=None))
    outputs = FSL2Scheme.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value