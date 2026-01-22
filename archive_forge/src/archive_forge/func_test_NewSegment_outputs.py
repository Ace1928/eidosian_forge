from ..preprocess import NewSegment
def test_NewSegment_outputs():
    output_map = dict(bias_corrected_images=dict(), bias_field_images=dict(), dartel_input_images=dict(), forward_deformation_field=dict(), inverse_deformation_field=dict(), modulated_class_images=dict(), native_class_images=dict(), normalized_class_images=dict(), transformation_mat=dict())
    outputs = NewSegment.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value