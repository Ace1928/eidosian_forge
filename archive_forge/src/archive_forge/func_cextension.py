import functools
def cextension(func):
    if ase_ext is None:
        return func
    cfunc = getattr(ase_ext, func.__name__, None)
    if cfunc is None:
        return func
    functools.update_wrapper(cfunc, func)
    cfunc.__pure_python_function__ = func
    return cfunc