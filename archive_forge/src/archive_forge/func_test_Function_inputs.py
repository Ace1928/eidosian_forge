from ..wrappers import Function
def test_Function_inputs():
    input_map = dict(function_str=dict(mandatory=True))
    inputs = Function.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value