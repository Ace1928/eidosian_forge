import numpy as np
import pytest
from opt_einsum import (backends, contract, contract_expression, helpers, sharing)
from opt_einsum.contract import Shaped, infer_backend, parse_backend
@pytest.mark.skipif(not found_theano, reason='Theano not installed.')
@pytest.mark.parametrize('string', tests)
def test_theano_with_sharing(string):
    views = helpers.build_views(string)
    ein = contract(string, *views, optimize=False, use_blas=False)
    shps = [v.shape for v in views]
    expr = contract_expression(string, *shps, optimize=True)
    with sharing.shared_intermediates() as cache:
        thn1 = expr(*views, backend='theano')
        assert sharing.get_sharing_cache() is cache
        cache_sz = len(cache)
        assert cache_sz > 0
        thn2 = expr(*views, backend='theano')
        assert len(cache) == cache_sz
    assert all((isinstance(t, theano.tensor.TensorVariable) for t in cache.values()))
    assert np.allclose(ein, thn1)
    assert np.allclose(ein, thn2)