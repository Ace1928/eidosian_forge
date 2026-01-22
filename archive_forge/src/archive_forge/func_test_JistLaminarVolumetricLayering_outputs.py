from ..developer import JistLaminarVolumetricLayering
def test_JistLaminarVolumetricLayering_outputs():
    output_map = dict(outContinuous=dict(extensions=None), outDiscrete=dict(extensions=None), outLayer=dict(extensions=None))
    outputs = JistLaminarVolumetricLayering.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value