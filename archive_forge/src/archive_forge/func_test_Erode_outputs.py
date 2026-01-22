from ..preprocess import Erode
def test_Erode_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = Erode.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value