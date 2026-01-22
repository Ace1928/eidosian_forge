from ..preprocess import TShift
def test_TShift_outputs():
    output_map = dict(out_file=dict(extensions=None), timing_file=dict(extensions=None))
    outputs = TShift.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value