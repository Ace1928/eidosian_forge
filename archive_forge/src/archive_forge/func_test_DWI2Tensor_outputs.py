from ..preprocess import DWI2Tensor
def test_DWI2Tensor_outputs():
    output_map = dict(tensor=dict(extensions=None))
    outputs = DWI2Tensor.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value