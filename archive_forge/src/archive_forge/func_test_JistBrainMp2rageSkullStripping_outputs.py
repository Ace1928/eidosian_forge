from ..developer import JistBrainMp2rageSkullStripping
def test_JistBrainMp2rageSkullStripping_outputs():
    output_map = dict(outBrain=dict(extensions=None), outMasked=dict(extensions=None), outMasked2=dict(extensions=None), outMasked3=dict(extensions=None))
    outputs = JistBrainMp2rageSkullStripping.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value