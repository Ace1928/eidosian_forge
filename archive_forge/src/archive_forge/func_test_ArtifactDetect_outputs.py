from ..rapidart import ArtifactDetect
def test_ArtifactDetect_outputs():
    output_map = dict(displacement_files=dict(), intensity_files=dict(), mask_files=dict(), norm_files=dict(), outlier_files=dict(), plot_files=dict(), statistic_files=dict())
    outputs = ArtifactDetect.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value