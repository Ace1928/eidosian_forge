from contextlib import contextmanager
@contextmanager
def obj_in_module(module, name, obj):
    backup_obj = getattr(module, name, None)
    setattr(module, name, obj)
    try:
        yield
    finally:
        if backup_obj:
            setattr(module, name, backup_obj)