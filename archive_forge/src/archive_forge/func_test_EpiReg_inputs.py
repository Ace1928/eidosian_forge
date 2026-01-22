from ..epi import EpiReg
def test_EpiReg_inputs():
    input_map = dict(args=dict(argstr='%s'), echospacing=dict(argstr='--echospacing=%f'), environ=dict(nohash=True, usedefault=True), epi=dict(argstr='--epi=%s', extensions=None, mandatory=True, position=-4), fmap=dict(argstr='--fmap=%s', extensions=None), fmapmag=dict(argstr='--fmapmag=%s', extensions=None), fmapmagbrain=dict(argstr='--fmapmagbrain=%s', extensions=None), no_clean=dict(argstr='--noclean', usedefault=True), no_fmapreg=dict(argstr='--nofmapreg'), out_base=dict(argstr='--out=%s', position=-1, usedefault=True), output_type=dict(), pedir=dict(argstr='--pedir=%s'), t1_brain=dict(argstr='--t1brain=%s', extensions=None, mandatory=True, position=-2), t1_head=dict(argstr='--t1=%s', extensions=None, mandatory=True, position=-3), weight_image=dict(argstr='--weight=%s', extensions=None), wmseg=dict(argstr='--wmseg=%s', extensions=None))
    inputs = EpiReg.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value