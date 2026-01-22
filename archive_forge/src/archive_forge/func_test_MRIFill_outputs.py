from ..utils import MRIFill
def test_MRIFill_outputs():
    output_map = dict(log_file=dict(extensions=None), out_file=dict(extensions=None))
    outputs = MRIFill.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value