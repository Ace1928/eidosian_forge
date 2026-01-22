import itertools
import sys
import numpy as np
import pytest
import opt_einsum as oe
def test_memory_paths():
    expression = 'abc,bdef,fghj,cem,mhk,ljk->adgl'
    views = oe.helpers.build_views(expression)
    path_ret = oe.contract_path(expression, *views, optimize='optimal', memory_limit=5)
    assert check_path(path_ret[0], [(0, 1, 2, 3, 4, 5)])
    path_ret = oe.contract_path(expression, *views, optimize='greedy', memory_limit=5)
    assert check_path(path_ret[0], [(0, 1, 2, 3, 4, 5)])
    path_ret = oe.contract_path(expression, *views, optimize='optimal', memory_limit=-1)
    assert check_path(path_ret[0], [(0, 3), (0, 4), (0, 2), (0, 2), (0, 1)])
    path_ret = oe.contract_path(expression, *views, optimize='greedy', memory_limit=-1)
    assert check_path(path_ret[0], [(0, 3), (0, 4), (0, 2), (0, 2), (0, 1)])