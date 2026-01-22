from ..misc import AddNoise
def test_AddNoise_inputs():
    input_map = dict(bg_dist=dict(mandatory=True, usedefault=True), dist=dict(mandatory=True, usedefault=True), in_file=dict(extensions=None, mandatory=True), in_mask=dict(extensions=None), out_file=dict(extensions=None), snr=dict(usedefault=True))
    inputs = AddNoise.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value