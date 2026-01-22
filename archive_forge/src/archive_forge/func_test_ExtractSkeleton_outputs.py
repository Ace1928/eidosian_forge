from ..extractskeleton import ExtractSkeleton
def test_ExtractSkeleton_outputs():
    output_map = dict(OutputImageFileName=dict(extensions=None, position=-1))
    outputs = ExtractSkeleton.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value