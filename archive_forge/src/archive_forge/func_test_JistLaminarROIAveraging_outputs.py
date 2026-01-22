from ..developer import JistLaminarROIAveraging
def test_JistLaminarROIAveraging_outputs():
    output_map = dict(outROI3=dict(extensions=None))
    outputs = JistLaminarROIAveraging.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value