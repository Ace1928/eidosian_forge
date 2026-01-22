from ..utils import ResampleImageBySpacing
def test_ResampleImageBySpacing_outputs():
    output_map = dict(output_image=dict(extensions=None))
    outputs = ResampleImageBySpacing.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value