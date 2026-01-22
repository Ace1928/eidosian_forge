from collections import defaultdict
import importlib
from numba.core import types, itanium_mangler
from numba.core.utils import _dynamic_modname, _dynamic_module

        Build a FunctionDescriptor for an object mode variant of a Python
        function.
        