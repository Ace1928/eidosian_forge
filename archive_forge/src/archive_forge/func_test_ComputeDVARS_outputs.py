from ..confounds import ComputeDVARS
def test_ComputeDVARS_outputs():
    output_map = dict(avg_nstd=dict(), avg_std=dict(), avg_vxstd=dict(), fig_nstd=dict(extensions=None), fig_std=dict(extensions=None), fig_vxstd=dict(extensions=None), out_all=dict(extensions=None), out_nstd=dict(extensions=None), out_std=dict(extensions=None), out_vxstd=dict(extensions=None))
    outputs = ComputeDVARS.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value