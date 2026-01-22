from ..preprocess import MultiChannelNewSegment
def test_MultiChannelNewSegment_inputs():
    input_map = dict(affine_regularization=dict(field='warp.affreg'), channels=dict(field='channel'), matlab_cmd=dict(), mfile=dict(usedefault=True), paths=dict(), sampling_distance=dict(field='warp.samp'), tissues=dict(field='tissue'), use_mcr=dict(), use_v8struct=dict(min_ver='8', usedefault=True), warping_regularization=dict(field='warp.reg'), write_deformation_fields=dict(field='warp.write'))
    inputs = MultiChannelNewSegment.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value