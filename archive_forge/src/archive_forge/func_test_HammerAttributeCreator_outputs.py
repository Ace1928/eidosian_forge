from ..featuredetection import HammerAttributeCreator
def test_HammerAttributeCreator_outputs():
    output_map = dict()
    outputs = HammerAttributeCreator.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value