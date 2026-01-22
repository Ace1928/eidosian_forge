from ..misc import NormalizeProbabilityMapSet
def test_NormalizeProbabilityMapSet_inputs():
    input_map = dict(in_files=dict(), in_mask=dict(extensions=None))
    inputs = NormalizeProbabilityMapSet.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value