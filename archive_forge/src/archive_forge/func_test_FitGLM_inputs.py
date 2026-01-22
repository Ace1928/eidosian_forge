from ..model import FitGLM
def test_FitGLM_inputs():
    input_map = dict(TR=dict(mandatory=True), drift_model=dict(usedefault=True), hrf_model=dict(usedefault=True), mask=dict(extensions=None), method=dict(usedefault=True), model=dict(usedefault=True), normalize_design_matrix=dict(usedefault=True), plot_design_matrix=dict(usedefault=True), save_residuals=dict(usedefault=True), session_info=dict(mandatory=True))
    inputs = FitGLM.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value