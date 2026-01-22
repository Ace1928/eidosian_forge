from ..misc import CreateNifti
def test_CreateNifti_inputs():
    input_map = dict(affine=dict(), data_file=dict(extensions=None, mandatory=True), header_file=dict(extensions=None, mandatory=True))
    inputs = CreateNifti.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value