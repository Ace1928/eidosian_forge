from ..preprocess import Tensor2Vector
def test_Tensor2Vector_outputs():
    output_map = dict(vector=dict(extensions=None))
    outputs = Tensor2Vector.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value