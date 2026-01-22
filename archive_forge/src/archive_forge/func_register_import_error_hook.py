import importlib  # noqa: F401
import sys
import threading
@synchronized(_import_error_hooks_lock)
def register_import_error_hook(hook, name, overwrite=True):
    """
    Args:
        hook: A function or string entrypoint to invoke when the specified module is imported
            and an error occurs.
        name: The name of the module for which to fire the hook at import error detection time.
        overwrite: Specifies the desired behavior when a preexisting hook for the same
            function / entrypoint already exists for the specified module. If `True`,
            all preexisting hooks matching the specified function / entrypoint will be
            removed and replaced with a single instance of the specified `hook`.
    """
    register_generic_import_hook(hook, name, _import_error_hooks, overwrite)