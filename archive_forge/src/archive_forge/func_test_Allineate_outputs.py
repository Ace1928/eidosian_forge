from ..preprocess import Allineate
def test_Allineate_outputs():
    output_map = dict(allcostx=dict(extensions=None), out_file=dict(extensions=None), out_matrix=dict(extensions=None), out_param_file=dict(extensions=None), out_weight_file=dict(extensions=None))
    outputs = Allineate.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value