def try_import_module(module_name):
    """
    Imports a module, but catches import errors.  Only catches errors
    when that module doesn't exist; if that module itself has an
    import error it will still get raised.  Returns None if the module
    doesn't exist.
    """
    try:
        return import_module(module_name)
    except ImportError as e:
        if not getattr(e, 'args', None):
            raise
        desc = e.args[0]
        if not desc.startswith('No module named '):
            raise
        desc = desc[len('No module named '):]
        parts = module_name.split('.')
        for i in range(len(parts)):
            if desc == '.'.join(parts[i:]):
                return None
        raise