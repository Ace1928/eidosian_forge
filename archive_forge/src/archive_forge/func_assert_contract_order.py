import itertools
import sys
import numpy as np
import pytest
import opt_einsum as oe
def assert_contract_order(func, test_data, max_size, benchmark):
    test_output = func(test_data[0], test_data[1], test_data[2], max_size)
    assert check_path(test_output, benchmark)