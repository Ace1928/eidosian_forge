import importlib  # noqa: F401
import sys
import threading
def discover_post_import_hooks(group):
    if sys.version_info.major > 2 and sys.version_info.minor > 8:
        from importlib.resources import files
        for entrypoint in (resource.name for resource in files(group).iterdir() if resource.is_file()):
            callback = _create_import_hook_from_entrypoint(entrypoint)
            register_post_import_hook(callback, entrypoint.name)
    else:
        from importlib.resources import contents
        for entrypoint in contents(group):
            callback = _create_import_hook_from_entrypoint(entrypoint)
            register_post_import_hook(callback, entrypoint.name)