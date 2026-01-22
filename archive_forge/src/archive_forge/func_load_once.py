import functools
import threading
import types
@_memoize
def load_once(self):
    if load_once.loading:
        raise ImportError('Circular import when resolving LazyModule %r' % name)
    load_once.loading = True
    try:
        module = load_fn()
    finally:
        load_once.loading = False
    self.__dict__.update(module.__dict__)
    load_once.loaded = True
    return module