from ..preprocess import ResponseSD
def test_ResponseSD_outputs():
    output_map = dict(csf_file=dict(argstr='%s', extensions=None), gm_file=dict(argstr='%s', extensions=None), wm_file=dict(argstr='%s', extensions=None))
    outputs = ResponseSD.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value