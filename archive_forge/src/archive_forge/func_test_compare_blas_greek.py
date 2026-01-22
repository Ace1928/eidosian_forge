import numpy as np
import pytest
from opt_einsum import contract, contract_expression, contract_path, helpers
from opt_einsum.paths import linear_to_ssa, ssa_to_linear
@pytest.mark.parametrize('string', tests)
@pytest.mark.parametrize('optimize', all_optimizers)
def test_compare_blas_greek(optimize, string):
    views = helpers.build_views(string)
    ein = contract(string, *views, optimize=False)
    string = ''.join((chr(ord(c) + 848) if c not in ',->.' else c for c in string))
    opt = contract(string, *views, optimize=optimize)
    assert np.allclose(ein, opt)