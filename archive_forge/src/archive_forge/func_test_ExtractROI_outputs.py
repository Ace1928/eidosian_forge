from ..utils import ExtractROI
def test_ExtractROI_outputs():
    output_map = dict(roi_file=dict(extensions=None))
    outputs = ExtractROI.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value