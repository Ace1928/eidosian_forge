from ..brains import BRAINSConstellationModeler
def test_BRAINSConstellationModeler_outputs():
    output_map = dict(outputModel=dict(extensions=None), resultsDir=dict())
    outputs = BRAINSConstellationModeler.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value