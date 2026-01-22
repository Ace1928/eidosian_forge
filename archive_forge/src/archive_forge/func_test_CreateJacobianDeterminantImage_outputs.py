from ..utils import CreateJacobianDeterminantImage
def test_CreateJacobianDeterminantImage_outputs():
    output_map = dict(jacobian_image=dict(extensions=None))
    outputs = CreateJacobianDeterminantImage.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value