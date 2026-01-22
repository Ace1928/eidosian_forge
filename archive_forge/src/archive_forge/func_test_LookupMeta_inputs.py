from ..dcmstack import LookupMeta
def test_LookupMeta_inputs():
    input_map = dict(in_file=dict(extensions=None, mandatory=True), meta_keys=dict(mandatory=True))
    inputs = LookupMeta.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value