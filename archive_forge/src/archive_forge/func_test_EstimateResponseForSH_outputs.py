from ..tensors import EstimateResponseForSH
def test_EstimateResponseForSH_outputs():
    output_map = dict(response=dict(extensions=None))
    outputs = EstimateResponseForSH.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value