from ..developer import JistBrainMp2rageDuraEstimation
def test_JistBrainMp2rageDuraEstimation_outputs():
    output_map = dict(outDura=dict(extensions=None))
    outputs = JistBrainMp2rageDuraEstimation.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value