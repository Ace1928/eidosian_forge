import inspect
from pyomo.common.deprecation import relocated_module_attribute
from pyomo.core.base.indexed_component import normalize_index
def flatten_tuple(x):
    """
    This wraps around normalize_index. It flattens a nested sequence into
    a single tuple and always returns a tuple, even for single
    element inputs.

    Returns
    -------
    tuple

    """
    x = normalize_index(x)
    if isinstance(x, tuple):
        return x
    return (x,)