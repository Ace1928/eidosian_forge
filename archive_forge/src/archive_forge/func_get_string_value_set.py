from jedi.inference.compiled.value import CompiledValue, CompiledName, \
from jedi.inference.base_value import LazyValueWrapper
def get_string_value_set(inference_state):
    return builtin_from_name(inference_state, 'str').execute_with_values()