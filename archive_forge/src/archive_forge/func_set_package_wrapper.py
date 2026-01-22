from ._abc import Loader
from ._bootstrap import module_from_spec
from ._bootstrap import _resolve_name
from ._bootstrap import spec_from_loader
from ._bootstrap import _find_spec
from ._bootstrap_external import MAGIC_NUMBER
from ._bootstrap_external import _RAW_MAGIC_NUMBER
from ._bootstrap_external import cache_from_source
from ._bootstrap_external import decode_source
from ._bootstrap_external import source_from_cache
from ._bootstrap_external import spec_from_file_location
from contextlib import contextmanager
import _imp
import functools
import sys
import types
import warnings
@functools.wraps(fxn)
def set_package_wrapper(*args, **kwargs):
    warnings.warn('The import system now takes care of this automatically; this decorator is slated for removal in Python 3.12', DeprecationWarning, stacklevel=2)
    module = fxn(*args, **kwargs)
    if getattr(module, '__package__', None) is None:
        module.__package__ = module.__name__
        if not hasattr(module, '__path__'):
            module.__package__ = module.__package__.rpartition('.')[0]
    return module