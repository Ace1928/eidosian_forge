from ..developer import JistLaminarProfileSampling
def test_JistLaminarProfileSampling_outputs():
    output_map = dict(outProfile2=dict(extensions=None), outProfilemapped=dict(extensions=None))
    outputs = JistLaminarProfileSampling.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value