from ..utils import ApplyTransform
def test_ApplyTransform_inputs():
    input_map = dict(in_file=dict(copyfile=True, extensions=None, mandatory=True), mat=dict(extensions=None, mandatory=True), matlab_cmd=dict(), mfile=dict(usedefault=True), out_file=dict(extensions=None, genfile=True), paths=dict(), use_mcr=dict(), use_v8struct=dict(min_ver='8', usedefault=True))
    inputs = ApplyTransform.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value