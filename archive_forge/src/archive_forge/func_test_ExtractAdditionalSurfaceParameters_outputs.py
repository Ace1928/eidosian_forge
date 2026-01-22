from ..surface import ExtractAdditionalSurfaceParameters
def test_ExtractAdditionalSurfaceParameters_outputs():
    output_map = dict(lh_area=dict(), lh_depth=dict(), lh_extracted_files=dict(), lh_fractaldimension=dict(), lh_gmv=dict(), lh_gyrification=dict(), rh_area=dict(), rh_depth=dict(), rh_extracted_files=dict(), rh_fractaldimension=dict(), rh_gmv=dict(), rh_gyrification=dict())
    outputs = ExtractAdditionalSurfaceParameters.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value