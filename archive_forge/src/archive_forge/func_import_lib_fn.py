import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def import_lib_fn(backend, fn):
    if fn in _COMPOSED_FUNCTION_GENERATORS:
        return _COMPOSED_FUNCTION_GENERATORS[fn](backend)
    try:
        try:
            full_location = _SUBMODULE_ALIASES[backend, fn]
            only_fn = fn.split('.')[-1]
        except KeyError:
            full_location = backend
            split_fn = fn.split('.')
            full_location = '.'.join([full_location] + split_fn[:-1])
            only_fn = split_fn[-1]
            for k, v in _MODULE_ALIASES.items():
                if full_location[:len(k)] == k:
                    full_location = full_location.replace(k, v, 1)
                    break
        fn_name = _FUNC_ALIASES.get((backend, fn), only_fn)
        try:
            lib = importlib.import_module(full_location)
        except ImportError:
            if '.' in full_location:
                mod, *submods = full_location.split('.')
                lib = importlib.import_module(mod)
                for submod in submods:
                    lib = getattr(lib, submod)
            else:
                raise AttributeError
        wrapper = _CUSTOM_WRAPPERS.get((backend, fn), lambda fn: fn)
        lib_fn = _FUNCS[backend, fn] = wrapper(getattr(lib, fn_name))
    except AttributeError:
        backend_alt = backend + '[alt]'
        if backend_alt in _MODULE_ALIASES:
            return import_lib_fn(backend_alt, fn)
        raise ImportError(f"autoray couldn't find function '{fn}' for backend '{backend.replace('[alt]', '')}'.")
    return lib_fn