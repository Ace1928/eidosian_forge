from ..converters import OrientScalarVolume
def test_OrientScalarVolume_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), inputVolume1=dict(argstr='%s', extensions=None, position=-2), orientation=dict(argstr='--orientation %s'), outputVolume=dict(argstr='%s', hash_files=False, position=-1))
    inputs = OrientScalarVolume.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value