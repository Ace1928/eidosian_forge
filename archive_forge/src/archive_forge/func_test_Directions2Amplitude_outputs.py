from ..tensors import Directions2Amplitude
def test_Directions2Amplitude_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = Directions2Amplitude.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value