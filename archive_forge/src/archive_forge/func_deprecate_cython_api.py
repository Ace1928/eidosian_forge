from inspect import Parameter, signature
import functools
import warnings
from importlib import import_module
def deprecate_cython_api(module, routine_name, new_name=None, message=None):
    """
    Deprecate an exported cdef function in a public Cython API module.

    Only functions can be deprecated; typedefs etc. cannot.

    Parameters
    ----------
    module : module
        Public Cython API module (e.g. scipy.linalg.cython_blas).
    routine_name : str
        Name of the routine to deprecate. May also be a fused-type
        routine (in which case its all specializations are deprecated).
    new_name : str
        New name to include in the deprecation warning message
    message : str
        Additional text in the deprecation warning message

    Examples
    --------
    Usually, this function would be used in the top-level of the
    module ``.pyx`` file:

    >>> from scipy._lib.deprecation import deprecate_cython_api
    >>> import scipy.linalg.cython_blas as mod
    >>> deprecate_cython_api(mod, "dgemm", "dgemm_new",
    ...                      message="Deprecated in Scipy 1.5.0")
    >>> del deprecate_cython_api, mod

    After this, Cython modules that use the deprecated function emit a
    deprecation warning when they are imported.

    """
    old_name = f'{module.__name__}.{routine_name}'
    if new_name is None:
        depdoc = '`%s` is deprecated!' % old_name
    else:
        depdoc = f'`{old_name}` is deprecated, use `{new_name}` instead!'
    if message is not None:
        depdoc += '\n' + message
    d = module.__pyx_capi__
    j = 0
    has_fused = False
    while True:
        fused_name = f'__pyx_fuse_{j}{routine_name}'
        if fused_name in d:
            has_fused = True
            d[_DeprecationHelperStr(fused_name, depdoc)] = d.pop(fused_name)
            j += 1
        else:
            break
    if not has_fused:
        d[_DeprecationHelperStr(routine_name, depdoc)] = d.pop(routine_name)