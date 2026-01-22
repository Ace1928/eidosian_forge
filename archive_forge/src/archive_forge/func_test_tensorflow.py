import numpy as np
import pytest
from opt_einsum import (backends, contract, contract_expression, helpers, sharing)
from opt_einsum.contract import Shaped, infer_backend, parse_backend
@pytest.mark.skipif(not found_tensorflow, reason='Tensorflow not installed.')
@pytest.mark.parametrize('string', tests)
def test_tensorflow(string):
    views = helpers.build_views(string)
    ein = contract(string, *views, optimize=False, use_blas=False)
    opt = np.empty_like(ein)
    shps = [v.shape for v in views]
    expr = contract_expression(string, *shps, optimize=True)
    sess = tf.Session(config=_TF_CONFIG)
    with sess.as_default():
        expr(*views, backend='tensorflow', out=opt)
    sess.close()
    assert np.allclose(ein, opt)
    tensorflow_views = [backends.to_tensorflow(view) for view in views]
    expr(*tensorflow_views)