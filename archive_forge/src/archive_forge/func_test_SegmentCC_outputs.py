from ..preprocess import SegmentCC
def test_SegmentCC_outputs():
    output_map = dict(out_file=dict(extensions=None), out_rotation=dict(extensions=None))
    outputs = SegmentCC.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value