from jedi import debug
from jedi import settings
from jedi.inference import recursion
from jedi.inference.base_value import ValueSet, NO_VALUES, HelperValueMixin, \
from jedi.inference.lazy_value import LazyKnownValues
from jedi.inference.helpers import infer_call_of_leaf
from jedi.inference.cache import inference_state_method_cache
def check_array_additions(context, sequence):
    """ Just a mapper function for the internal _internal_check_array_additions """
    if sequence.array_type not in ('list', 'set'):
        return NO_VALUES
    return _internal_check_array_additions(context, sequence)