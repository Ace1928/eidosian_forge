from ..preprocess import MRTrixViewer
def test_MRTrixViewer_inputs():
    input_map = dict(args=dict(argstr='%s'), debug=dict(argstr='-debug', position=1), environ=dict(nohash=True, usedefault=True), in_files=dict(argstr='%s', mandatory=True, position=-2), quiet=dict(argstr='-quiet', position=1))
    inputs = MRTrixViewer.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value