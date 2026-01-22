from __future__ import annotations
import sys
import eventlet
def patch_function(func, *additional_modules):
    """Decorator that returns a version of the function that patches
    some modules for the duration of the function call.  This is
    deeply gross and should only be used for functions that import
    network libraries within their function bodies that there is no
    way of getting around."""
    if not additional_modules:
        additional_modules = _green_os_modules() + _green_select_modules() + _green_socket_modules() + _green_thread_modules() + _green_time_modules()

    def patched(*args, **kw):
        saver = SysModulesSaver()
        for name, mod in additional_modules:
            saver.save(name)
            sys.modules[name] = mod
        try:
            return func(*args, **kw)
        finally:
            saver.restore()
    return patched