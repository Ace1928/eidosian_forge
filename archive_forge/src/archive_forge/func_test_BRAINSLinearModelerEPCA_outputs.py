from ..brains import BRAINSLinearModelerEPCA
def test_BRAINSLinearModelerEPCA_outputs():
    output_map = dict()
    outputs = BRAINSLinearModelerEPCA.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value