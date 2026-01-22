from ..utils import LTAConvert
def test_LTAConvert_outputs():
    output_map = dict(out_fsl=dict(extensions=None), out_itk=dict(extensions=None), out_lta=dict(extensions=None), out_mni=dict(extensions=None), out_reg=dict(extensions=None))
    outputs = LTAConvert.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value