from ..reconst import ConstrainedSphericalDeconvolution
def test_ConstrainedSphericalDeconvolution_outputs():
    output_map = dict(csf_odf=dict(argstr='%s', extensions=None), gm_odf=dict(argstr='%s', extensions=None), predicted_signal=dict(extensions=None), wm_odf=dict(argstr='%s', extensions=None))
    outputs = ConstrainedSphericalDeconvolution.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value