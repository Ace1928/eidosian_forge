from collections import defaultdict
import importlib
from numba.core import types, itanium_mangler
from numba.core.utils import _dynamic_modname, _dynamic_module
def qualifying_prefix(modname, qualname):
    """
    Returns a new string that is used for the first half of the mangled name.
    """
    return '{}.{}'.format(modname, qualname) if modname else qualname