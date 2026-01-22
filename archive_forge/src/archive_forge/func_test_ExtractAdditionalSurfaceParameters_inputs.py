from ..surface import ExtractAdditionalSurfaceParameters
def test_ExtractAdditionalSurfaceParameters_inputs():
    input_map = dict(area=dict(field='area', usedefault=True), depth=dict(field='SD', usedefault=True), fractal_dimension=dict(field='FD', usedefault=True), gmv=dict(field='gmv', usedefault=True), gyrification=dict(field='GI', usedefault=True), left_central_surfaces=dict(copyfile=False, field='data_surf', mandatory=True), matlab_cmd=dict(), mfile=dict(usedefault=True), paths=dict(), surface_files=dict(copyfile=False, mandatory=False), use_mcr=dict(), use_v8struct=dict(min_ver='8', usedefault=True))
    inputs = ExtractAdditionalSurfaceParameters.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value