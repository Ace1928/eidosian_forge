import importlib  # noqa: F401
import sys
import threading
def register_generic_import_hook(hook, name, hook_dict, overwrite):
    if isinstance(hook, string_types):
        hook = _create_import_hook_from_string(hook)
    global _import_hook_finder_init
    if not _import_hook_finder_init:
        _import_hook_finder_init = True
        sys.meta_path.insert(0, ImportHookFinder())
    hooks = hook_dict.get(name, None)
    if hooks is None:
        module = sys.modules.get(name, None)
        if module is not None:
            hook_dict[name] = []
            hook(module)
        else:
            hook_dict[name] = [hook]
    elif hooks == []:
        module = sys.modules[name]
        hook(module)
    else:

        def hooks_equal(existing_hook, hook):
            if hasattr(existing_hook, '__name__') and hasattr(hook, '__name__'):
                return existing_hook.__name__ == hook.__name__
            else:
                return False
        if overwrite:
            hook_dict[name] = [existing_hook for existing_hook in hook_dict[name] if not hooks_equal(existing_hook, hook)]
        hook_dict[name].append(hook)