from ..dti import TractSkeleton
def test_TractSkeleton_outputs():
    output_map = dict(projected_data=dict(extensions=None), skeleton_file=dict(extensions=None))
    outputs = TractSkeleton.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value