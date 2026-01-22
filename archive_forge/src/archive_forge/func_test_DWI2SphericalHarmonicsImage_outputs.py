from ..tensors import DWI2SphericalHarmonicsImage
def test_DWI2SphericalHarmonicsImage_outputs():
    output_map = dict(spherical_harmonics_image=dict(extensions=None))
    outputs = DWI2SphericalHarmonicsImage.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value