import functools
import types
import warnings
import importlib
import sys
from gi import PyGIDeprecationWarning
from gi._gi import CallableInfo, pygobject_new_full
from gi._constants import \
from pkgutil import extend_path
def load_overrides(introspection_module):
    """Loads overrides for an introspection module.

    Either returns the same module again in case there are no overrides or a
    proxy module including overrides. Doesn't cache the result.
    """
    namespace = introspection_module.__name__.rsplit('.', 1)[-1]
    module_key = 'gi.repository.' + namespace
    has_old = module_key in sys.modules
    old_module = sys.modules.get(module_key)
    proxy_type = type(namespace + 'ProxyModule', (OverridesProxyModule,), {})
    proxy = proxy_type(introspection_module)
    sys.modules[module_key] = proxy
    from ..importer import modules
    assert hasattr(proxy, '_introspection_module')
    modules[namespace] = proxy
    try:
        override_package_name = 'gi.overrides.' + namespace
        spec = importlib.util.find_spec(override_package_name)
        override_loader = spec.loader if spec is not None else None
        if override_loader is None:
            return introspection_module
        override_mod = importlib.import_module(override_package_name)
    finally:
        del modules[namespace]
        del sys.modules[module_key]
        if has_old:
            sys.modules[module_key] = old_module
    proxy._overrides_module = proxy
    override_all = []
    if hasattr(override_mod, '__all__'):
        override_all = override_mod.__all__
    for var in override_all:
        try:
            item = getattr(override_mod, var)
        except (AttributeError, TypeError):
            continue
        setattr(proxy, var, item)
    for attr, replacement in _deprecated_attrs.pop(namespace, []):
        try:
            value = getattr(proxy, attr)
        except AttributeError:
            raise AssertionError("%s was set deprecated but wasn't added to __all__" % attr)
        delattr(proxy, attr)
        deprecated_attr = _DeprecatedAttribute(namespace, attr, value, replacement)
        setattr(proxy_type, attr, deprecated_attr)
    return proxy