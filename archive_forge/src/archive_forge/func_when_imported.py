import importlib  # noqa: F401
import sys
import threading
def when_imported(name, error_handler=False):

    def register(hook):
        if error_handler:
            register_import_error_hook(hook, name)
        else:
            register_post_import_hook(hook, name)
        return hook
    return register