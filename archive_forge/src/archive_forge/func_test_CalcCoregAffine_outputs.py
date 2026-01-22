from ..utils import CalcCoregAffine
def test_CalcCoregAffine_outputs():
    output_map = dict(invmat=dict(extensions=None), mat=dict(extensions=None))
    outputs = CalcCoregAffine.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value