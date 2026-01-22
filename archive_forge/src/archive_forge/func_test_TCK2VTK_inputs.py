from ..utils import TCK2VTK
def test_TCK2VTK_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), in_file=dict(argstr='%s', extensions=None, mandatory=True, position=-2), nthreads=dict(argstr='-nthreads %d', nohash=True), out_file=dict(argstr='%s', extensions=None, position=-1, usedefault=True), reference=dict(argstr='-image %s', extensions=None), voxel=dict(argstr='-image %s', extensions=None))
    inputs = TCK2VTK.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value