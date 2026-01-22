from ..utils import GCOR
def test_GCOR_outputs():
    output_map = dict(out=dict())
    outputs = GCOR.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value