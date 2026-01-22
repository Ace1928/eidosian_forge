from collections import defaultdict
import importlib
from numba.core import types, itanium_mangler
from numba.core.utils import _dynamic_modname, _dynamic_module
def lookup_function(self):
    """
        Return the original function object described by this object.
        """
    return getattr(self.lookup_module(), self.qualname)