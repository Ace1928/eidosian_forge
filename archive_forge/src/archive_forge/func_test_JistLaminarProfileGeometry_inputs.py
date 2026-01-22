from ..developer import JistLaminarProfileGeometry
def test_JistLaminarProfileGeometry_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), inProfile=dict(argstr='--inProfile %s', extensions=None), incomputed=dict(argstr='--incomputed %s'), inoutside=dict(argstr='--inoutside %f'), inregularization=dict(argstr='--inregularization %s'), insmoothing=dict(argstr='--insmoothing %f'), null=dict(argstr='--null %s'), outResult=dict(argstr='--outResult %s', hash_files=False), xDefaultMem=dict(argstr='-xDefaultMem %d'), xMaxProcess=dict(argstr='-xMaxProcess %d', usedefault=True), xPrefExt=dict(argstr='--xPrefExt %s'))
    inputs = JistLaminarProfileGeometry.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value