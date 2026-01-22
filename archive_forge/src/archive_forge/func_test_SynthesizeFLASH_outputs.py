from ..preprocess import SynthesizeFLASH
def test_SynthesizeFLASH_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = SynthesizeFLASH.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value