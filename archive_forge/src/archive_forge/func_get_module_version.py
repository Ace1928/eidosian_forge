import sys
from packaging import version as _version
def get_module_version(module_name):
    """Return module version or None if version can't be retrieved."""
    mod = __import__(module_name, fromlist=[module_name.rpartition('.')[-1]])
    return getattr(mod, '__version__', getattr(mod, 'VERSION', None))