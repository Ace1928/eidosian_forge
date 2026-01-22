import itertools
import sys
import numpy as np
import pytest
import opt_einsum as oe
@pytest.mark.parametrize('alg,expression,order', path_edge_tests)
def test_path_edge_cases(alg, expression, order):
    views = oe.helpers.build_views(expression)
    path_ret = oe.contract_path(expression, *views, optimize=alg)
    assert check_path(path_ret[0], order)