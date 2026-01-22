from ..brainsuite import Cortex
def test_Cortex_outputs():
    output_map = dict(outputCerebrumMask=dict(extensions=None))
    outputs = Cortex.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value