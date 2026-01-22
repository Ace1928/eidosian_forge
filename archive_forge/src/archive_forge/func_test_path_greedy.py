import itertools
import sys
import numpy as np
import pytest
import opt_einsum as oe
def test_path_greedy():
    test_func = oe.paths.greedy
    test_data = explicit_path_tests['GEMM1']
    assert_contract_order(test_func, test_data, 5000, [(0, 2), (0, 1)])
    assert_contract_order(test_func, test_data, 0, [(0, 1, 2)])