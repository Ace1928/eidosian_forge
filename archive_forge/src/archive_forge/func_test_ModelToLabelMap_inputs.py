from ..surface import ModelToLabelMap
def test_ModelToLabelMap_inputs():
    input_map = dict(InputVolume=dict(argstr='%s', extensions=None, position=-3), OutputVolume=dict(argstr='%s', hash_files=False, position=-1), args=dict(argstr='%s'), distance=dict(argstr='--distance %f'), environ=dict(nohash=True, usedefault=True), surface=dict(argstr='%s', extensions=None, position=-2))
    inputs = ModelToLabelMap.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value