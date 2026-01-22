from ..preprocess import Automask
def test_Automask_outputs():
    output_map = dict(brain_file=dict(extensions=None), out_file=dict(extensions=None))
    outputs = Automask.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value