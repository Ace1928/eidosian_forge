from ..utils import EditTransform
def test_EditTransform_inputs():
    input_map = dict(interpolation=dict(argstr='FinalBSplineInterpolationOrder', usedefault=True), output_file=dict(extensions=None), output_format=dict(argstr='ResultImageFormat'), output_type=dict(argstr='ResultImagePixelType'), reference_image=dict(extensions=None), transform_file=dict(extensions=None, mandatory=True))
    inputs = EditTransform.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value