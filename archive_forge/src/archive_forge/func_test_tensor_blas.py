import numpy as np
import pytest
from opt_einsum import blas, contract, helpers
@pytest.mark.parametrize('inp,benchmark', blas_tests)
def test_tensor_blas(inp, benchmark):
    if benchmark is False:
        return
    tensor_strs, output, reduced_idx = inp
    einsum_str = ','.join(tensor_strs) + '->' + output
    if len(tensor_strs) != 2:
        assert False
    view_left, view_right = helpers.build_views(einsum_str)
    einsum_result = np.einsum(einsum_str, view_left, view_right)
    blas_result = blas.tensor_blas(view_left, tensor_strs[0], view_right, tensor_strs[1], output, reduced_idx)
    assert np.allclose(einsum_result, blas_result)