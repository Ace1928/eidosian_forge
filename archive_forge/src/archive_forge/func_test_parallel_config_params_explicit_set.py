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
@with_numpy
@with_multiprocessing
def test_parallel_config_params_explicit_set(tmpdir):
    with parallel_config(n_jobs=3, max_nbytes=1, temp_folder=tmpdir):
        with Parallel(n_jobs=2, prefer='processes', max_nbytes='1M') as p:
            assert isinstance(p._backend, LokyBackend)
            assert p.n_jobs == 2
            with raises(TypeError, match='Expected np.memmap instance'):
                p((delayed(check_memmap)(a) for a in [np.random.random(10)] * 2))