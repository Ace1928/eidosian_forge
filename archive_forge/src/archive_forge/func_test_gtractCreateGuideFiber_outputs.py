from ..gtract import gtractCreateGuideFiber
def test_gtractCreateGuideFiber_outputs():
    output_map = dict(outputFiber=dict(extensions=None))
    outputs = gtractCreateGuideFiber.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value