from ..registration import MPRtoMNI305
def test_MPRtoMNI305_outputs():
    output_map = dict(log_file=dict(extensions=None, usedefault=True), out_file=dict(extensions=None))
    outputs = MPRtoMNI305.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value