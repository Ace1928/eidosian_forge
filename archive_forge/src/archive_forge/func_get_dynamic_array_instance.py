from jedi import debug
from jedi import settings
from jedi.inference import recursion
from jedi.inference.base_value import ValueSet, NO_VALUES, HelperValueMixin, \
from jedi.inference.lazy_value import LazyKnownValues
from jedi.inference.helpers import infer_call_of_leaf
from jedi.inference.cache import inference_state_method_cache
def get_dynamic_array_instance(instance, arguments):
    """Used for set() and list() instances."""
    ai = _DynamicArrayAdditions(instance, arguments)
    from jedi.inference import arguments
    return arguments.ValuesArguments([ValueSet([ai])])