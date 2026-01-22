from ..brainsuite import Skullfinder
def test_Skullfinder_outputs():
    output_map = dict(outputLabelFile=dict(extensions=None))
    outputs = Skullfinder.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value