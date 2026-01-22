from ..brainsuite import Cortex
def test_Cortex_inputs():
    input_map = dict(args=dict(argstr='%s'), computeGCBoundary=dict(argstr='-g'), computeWGBoundary=dict(argstr='-w', usedefault=True), environ=dict(nohash=True, usedefault=True), includeAllSubcorticalAreas=dict(argstr='-a', usedefault=True), inputHemisphereLabelFile=dict(argstr='-h %s', extensions=None, mandatory=True), inputTissueFractionFile=dict(argstr='-f %s', extensions=None, mandatory=True), outputCerebrumMask=dict(argstr='-o %s', extensions=None, genfile=True), timer=dict(argstr='--timer'), tissueFractionThreshold=dict(argstr='-p %f', usedefault=True), verbosity=dict(argstr='-v %d'))
    inputs = Cortex.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value