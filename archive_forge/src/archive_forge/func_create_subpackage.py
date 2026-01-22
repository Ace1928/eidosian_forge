import functools
import importlib.util
import pkgutil
import sys
import types
from oslo_log import log as logging
def create_subpackage(path, parent_package_name, subpackage_name='plugins'):
    """Dynamically create a package into which to load plugins.

    This allows us to not include an __init__.py in the plugins directory. We
    must still create a package for plugins to go in, otherwise we get warning
    messages during import. This also provides a convenient place to store the
    path(s) to the plugins directory.
    """
    package_name = _module_name(parent_package_name, subpackage_name)
    package = types.ModuleType(package_name)
    package.__path__ = [path] if isinstance(path, str) else list(path)
    sys.modules[package_name] = package
    return package