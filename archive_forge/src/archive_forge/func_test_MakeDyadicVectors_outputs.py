from ..dti import MakeDyadicVectors
def test_MakeDyadicVectors_outputs():
    output_map = dict(dispersion=dict(extensions=None), dyads=dict(extensions=None))
    outputs = MakeDyadicVectors.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value