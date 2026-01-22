import contextlib
import os
import sklearn
from .._min_dependencies import CYTHON_MIN_VERSION
from ..externals._packaging.version import parse
from .openmp_helpers import check_openmp_support
from .pre_build_helpers import basic_check_build
def cythonize_extensions(extension):
    """Check that a recent Cython is available and cythonize extensions"""
    _check_cython_version()
    from Cython.Build import cythonize
    basic_check_build()
    sklearn._OPENMP_SUPPORTED = check_openmp_support()
    n_jobs = 1
    with contextlib.suppress(ImportError):
        import joblib
        n_jobs = joblib.cpu_count()
    cython_enable_debug_directives = os.environ.get('SKLEARN_ENABLE_DEBUG_CYTHON_DIRECTIVES', '0') != '0'
    compiler_directives = {'language_level': 3, 'boundscheck': cython_enable_debug_directives, 'wraparound': False, 'initializedcheck': False, 'nonecheck': False, 'cdivision': True, 'profile': False}
    return cythonize(extension, nthreads=n_jobs, compiler_directives=compiler_directives, annotate=False)