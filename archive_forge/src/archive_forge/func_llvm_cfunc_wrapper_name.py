from collections import defaultdict
import importlib
from numba.core import types, itanium_mangler
from numba.core.utils import _dynamic_modname, _dynamic_module
@property
def llvm_cfunc_wrapper_name(self):
    """
        The LLVM-registered name for a C-compatible wrapper of the
        raw function.
        """
    return 'cfunc.' + self.mangled_name