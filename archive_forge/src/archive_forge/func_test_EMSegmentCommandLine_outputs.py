from ..specialized import EMSegmentCommandLine
def test_EMSegmentCommandLine_outputs():
    output_map = dict(generateEmptyMRMLSceneAndQuit=dict(extensions=None), resultMRMLSceneFileName=dict(extensions=None), resultVolumeFileName=dict(extensions=None))
    outputs = EMSegmentCommandLine.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value