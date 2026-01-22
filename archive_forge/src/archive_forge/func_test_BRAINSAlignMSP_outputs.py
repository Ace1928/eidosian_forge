from ..brains import BRAINSAlignMSP
def test_BRAINSAlignMSP_outputs():
    output_map = dict(OutputresampleMSP=dict(extensions=None), resultsDir=dict())
    outputs = BRAINSAlignMSP.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value