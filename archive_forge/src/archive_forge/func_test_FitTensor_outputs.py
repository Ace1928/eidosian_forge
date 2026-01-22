from ..reconst import FitTensor
def test_FitTensor_outputs():
    output_map = dict(out_file=dict(extensions=None), predicted_signal=dict(extensions=None))
    outputs = FitTensor.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value