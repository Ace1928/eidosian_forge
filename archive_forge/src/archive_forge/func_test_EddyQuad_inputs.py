from ..epi import EddyQuad
def test_EddyQuad_inputs():
    input_map = dict(args=dict(argstr='%s'), base_name=dict(argstr='%s', position=0, usedefault=True), bval_file=dict(argstr='--bvals %s', extensions=None, mandatory=True), bvec_file=dict(argstr='--bvecs %s', extensions=None), environ=dict(nohash=True, usedefault=True), field=dict(argstr='--field %s', extensions=None), idx_file=dict(argstr='--eddyIdx %s', extensions=None, mandatory=True), mask_file=dict(argstr='--mask %s', extensions=None, mandatory=True), output_dir=dict(argstr='--output-dir %s', name_source=['base_name'], name_template='%s.qc'), output_type=dict(), param_file=dict(argstr='--eddyParams %s', extensions=None, mandatory=True), slice_spec=dict(argstr='--slspec %s', extensions=None), verbose=dict(argstr='--verbose'))
    inputs = EddyQuad.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value