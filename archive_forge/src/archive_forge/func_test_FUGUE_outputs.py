from ..preprocess import FUGUE
def test_FUGUE_outputs():
    output_map = dict(fmap_out_file=dict(extensions=None), shift_out_file=dict(extensions=None), unwarped_file=dict(extensions=None), warped_file=dict(extensions=None))
    outputs = FUGUE.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value