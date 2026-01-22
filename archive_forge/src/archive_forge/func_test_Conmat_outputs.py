from ..connectivity import Conmat
def test_Conmat_outputs():
    output_map = dict(conmat_sc=dict(extensions=None), conmat_ts=dict(extensions=None))
    outputs = Conmat.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value