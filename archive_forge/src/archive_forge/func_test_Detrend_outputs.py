from ..preprocess import Detrend
def test_Detrend_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = Detrend.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value