import os
from joblib.parallel import parallel_config
from joblib.parallel import parallel_backend
from joblib.parallel import Parallel, delayed
from joblib.parallel import BACKENDS
from joblib.parallel import DEFAULT_BACKEND
from joblib.parallel import EXTERNAL_BACKENDS
from joblib._parallel_backends import LokyBackend
from joblib._parallel_backends import ThreadingBackend
from joblib._parallel_backends import MultiprocessingBackend
from joblib.testing import parametrize, raises
from joblib.test.common import np, with_numpy
from joblib.test.common import with_multiprocessing
from joblib.test.test_parallel import check_memmap
def test_parallel_config_constructor_params():
    with raises(ValueError, match='only supported when backend is not None'):
        with parallel_config(inner_max_num_threads=1):
            pass
    with raises(ValueError, match='only supported when backend is not None'):
        with parallel_config(backend_param=1):
            pass