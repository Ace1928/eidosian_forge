from ..registration import BSplineDeformableRegistration
def test_BSplineDeformableRegistration_outputs():
    output_map = dict(outputtransform=dict(extensions=None), outputwarp=dict(extensions=None), resampledmovingfilename=dict(extensions=None))
    outputs = BSplineDeformableRegistration.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value