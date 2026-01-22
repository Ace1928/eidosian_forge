import numpy as np
import pytest
from opt_einsum import contract, contract_path
def test_large_int_input_format():
    string = 'ab,bc,cd'
    x, y, z = build_views(string)
    string_output = contract(string, x, y, z)
    int_output = contract(x, (1000, 1001), y, (1001, 1002), z, (1002, 1003))
    assert np.allclose(string_output, int_output)
    for i in range(10):
        transpose_output = contract(x, (i + 1, i))
        assert np.allclose(transpose_output, x.T)