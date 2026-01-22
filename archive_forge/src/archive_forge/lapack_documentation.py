import numpy as _np
from .blas import _get_funcs, _memoize_get_funcs
from scipy.linalg import _flapack
from re import compile as regex_compile
from scipy.linalg._flapack import *  # noqa: E402, F403

    Convert LAPACK-returned work array size float to integer,
    carefully for single-precision types.
    