from ..specialized import BRAINSCut
def test_BRAINSCut_outputs():
    output_map = dict()
    outputs = BRAINSCut.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value