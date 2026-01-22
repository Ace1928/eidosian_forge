from ..gtract import gtractTransformToDisplacementField
def test_gtractTransformToDisplacementField_outputs():
    output_map = dict(outputDeformationFieldVolume=dict(extensions=None))
    outputs = gtractTransformToDisplacementField.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value