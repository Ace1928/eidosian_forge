from ..petsurfer import GTMPVC
def test_GTMPVC_outputs():
    output_map = dict(gtm_file=dict(extensions=None), gtm_stats=dict(extensions=None), hb_dat=dict(extensions=None), hb_nifti=dict(extensions=None), input_file=dict(extensions=None), mgx_ctxgm=dict(extensions=None), mgx_gm=dict(extensions=None), mgx_subctxgm=dict(extensions=None), nopvc_file=dict(extensions=None), opt_params=dict(extensions=None), pvc_dir=dict(), rbv=dict(extensions=None), ref_file=dict(extensions=None), reg_anat2pet=dict(extensions=None), reg_anat2rbvpet=dict(extensions=None), reg_pet2anat=dict(extensions=None), reg_rbvpet2anat=dict(extensions=None), yhat=dict(extensions=None), yhat0=dict(extensions=None), yhat_full_fov=dict(extensions=None), yhat_with_noise=dict(extensions=None))
    outputs = GTMPVC.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value