from ..specialized import BRAINSABC
def test_BRAINSABC_outputs():
    output_map = dict(atlasToSubjectInitialTransform=dict(extensions=None), atlasToSubjectTransform=dict(extensions=None), implicitOutputs=dict(), outputDir=dict(), outputDirtyLabels=dict(extensions=None), outputLabels=dict(extensions=None), outputVolumes=dict(), saveState=dict(extensions=None))
    outputs = BRAINSABC.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value