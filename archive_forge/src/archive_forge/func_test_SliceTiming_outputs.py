from ..preprocess import SliceTiming
def test_SliceTiming_outputs():
    output_map = dict(timecorrected_files=dict())
    outputs = SliceTiming.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value