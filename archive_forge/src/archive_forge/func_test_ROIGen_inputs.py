from ..cmtk import ROIGen
def test_ROIGen_inputs():
    input_map = dict(LUT_file=dict(extensions=None, xor=['use_freesurfer_LUT']), aparc_aseg_file=dict(extensions=None, mandatory=True), freesurfer_dir=dict(requires=['use_freesurfer_LUT']), out_dict_file=dict(extensions=None, genfile=True), out_roi_file=dict(extensions=None, genfile=True), use_freesurfer_LUT=dict(xor=['LUT_file']))
    inputs = ROIGen.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value