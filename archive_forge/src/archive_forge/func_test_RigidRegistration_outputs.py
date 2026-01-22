from ..registration import RigidRegistration
def test_RigidRegistration_outputs():
    output_map = dict(outputtransform=dict(extensions=None), resampledmovingfilename=dict(extensions=None))
    outputs = RigidRegistration.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value