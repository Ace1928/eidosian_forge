import importlib
from threading import Lock
import gi
from ._gi import \
from .types import \
from ._constants import \
def get_introspection_module(namespace):
    """
    :Returns:
        An object directly wrapping the gi module without overrides.

    Might raise gi._gi.RepositoryError
    """
    if namespace in _introspection_modules:
        return _introspection_modules[namespace]
    version = gi.get_required_version(namespace)
    module = IntrospectionModule(namespace, version)
    _introspection_modules[namespace] = module
    return module