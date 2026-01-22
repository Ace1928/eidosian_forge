import importlib
import inspect
import os
import warnings
from eventlet import patcher
from eventlet.support import greenlets as greenlet
from eventlet import timeout
def use_hub(mod=None):
    """Use the module *mod*, containing a class called Hub, as the
    event hub. Usually not required; the default hub is usually fine.

    `mod` can be an actual hub class, a module, a string, or None.

    If `mod` is a class, use it directly.
    If `mod` is a module, use `module.Hub` class
    If `mod` is a string and contains either '.' or ':'
    then `use_hub` uses 'package.subpackage.module:Class' convention,
    otherwise imports `eventlet.hubs.mod`.
    If `mod` is None, `use_hub` uses the default hub.

    Only call use_hub during application initialization,
    because it resets the hub's state and any existing
    timers or listeners will never be resumed.

    These two threadlocal attributes are not part of Eventlet public API:
    - `threadlocal.Hub` (capital H) is hub constructor, used when no hub is currently active
    - `threadlocal.hub` (lowercase h) is active hub instance
    """
    if mod is None:
        mod = os.environ.get('EVENTLET_HUB', None)
    if mod is None:
        mod = get_default_hub()
    if hasattr(_threadlocal, 'hub'):
        del _threadlocal.hub
    classname = ''
    if isinstance(mod, str):
        if mod.strip() == '':
            raise RuntimeError('Need to specify a hub')
        if '.' in mod or ':' in mod:
            modulename, _, classname = mod.strip().partition(':')
        else:
            modulename = 'eventlet.hubs.' + mod
        mod = importlib.import_module(modulename)
    if hasattr(mod, 'is_available'):
        if not mod.is_available():
            raise Exception('selected hub is not available on this system mod={}'.format(mod))
    else:
        msg = 'Please provide `is_available()` function in your custom Eventlet hub {mod}.\nIt must return bool: whether hub supports current platform. See eventlet/hubs/{{epoll,kqueue}} for example.\n'.format(mod=mod)
        warnings.warn(msg, DeprecationWarning, stacklevel=3)
    hubclass = mod
    if not inspect.isclass(mod):
        hubclass = getattr(mod, classname or 'Hub')
    _threadlocal.Hub = hubclass