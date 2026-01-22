import numpy as np
import pytest
from opt_einsum import contract, contract_path
def test_hashable_object_input_format():
    string = 'ab,bc,cd'
    x, y, z = build_views(string)
    string_output = contract(string, x, y, z)
    hash_output1 = contract(x, ('left', 'bond1'), y, ('bond1', 'bond2'), z, ('bond2', 'right'))
    hash_output2 = contract(x, ('left', 'bond1'), y, ('bond1', 'bond2'), z, ('bond2', 'right'), ('left', 'right'))
    assert np.allclose(string_output, hash_output1)
    assert np.allclose(hash_output1, hash_output2)
    for i in range(1, 10):
        transpose_output = contract(x, ('b' * i, 'a' * i))
        assert np.allclose(transpose_output, x.T)