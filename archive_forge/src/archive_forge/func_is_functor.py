import inspect
from pyomo.common.deprecation import relocated_module_attribute
from pyomo.core.base.indexed_component import normalize_index
def is_functor(obj):
    """
    Returns true iff obj.__call__ is defined.
    """
    return inspect.isfunction(obj) or hasattr(obj, '__call__')