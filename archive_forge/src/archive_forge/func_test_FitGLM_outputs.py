from ..model import FitGLM
def test_FitGLM_outputs():
    output_map = dict(a=dict(extensions=None), axis=dict(), beta=dict(extensions=None), constants=dict(), dof=dict(), nvbeta=dict(), reg_names=dict(), residuals=dict(extensions=None), s2=dict(extensions=None))
    outputs = FitGLM.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value