from ..parcellation import Parcellate
def test_Parcellate_inputs():
    input_map = dict(dilation=dict(usedefault=True), freesurfer_dir=dict(), out_roi_file=dict(extensions=None, genfile=True), parcellation_name=dict(usedefault=True), subject_id=dict(mandatory=True), subjects_dir=dict())
    inputs = Parcellate.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value