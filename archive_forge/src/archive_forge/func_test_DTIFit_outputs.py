from ..dti import DTIFit
def test_DTIFit_outputs():
    output_map = dict(tensor_fitted=dict(extensions=None))
    outputs = DTIFit.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value