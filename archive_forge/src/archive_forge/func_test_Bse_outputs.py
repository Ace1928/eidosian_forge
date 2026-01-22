from ..brainsuite import Bse
def test_Bse_outputs():
    output_map = dict(outputCortexFile=dict(extensions=None), outputDetailedBrainMask=dict(extensions=None), outputDiffusionFilter=dict(extensions=None), outputEdgeMap=dict(extensions=None), outputMRIVolume=dict(extensions=None), outputMaskFile=dict(extensions=None))
    outputs = Bse.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value