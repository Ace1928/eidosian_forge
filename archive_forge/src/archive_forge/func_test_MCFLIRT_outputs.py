from ..preprocess import MCFLIRT
def test_MCFLIRT_outputs():
    output_map = dict(mat_file=dict(), mean_img=dict(extensions=None), out_file=dict(extensions=None), par_file=dict(extensions=None), rms_files=dict(), std_img=dict(extensions=None), variance_img=dict(extensions=None))
    outputs = MCFLIRT.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value