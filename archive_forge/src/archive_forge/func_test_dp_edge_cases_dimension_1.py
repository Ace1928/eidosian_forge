import itertools
import sys
import numpy as np
import pytest
import opt_einsum as oe
def test_dp_edge_cases_dimension_1():
    eq = 'nlp,nlq,pl->n'
    shapes = [(1, 1, 1), (1, 1, 1), (1, 1)]
    info = oe.contract_path(eq, *shapes, shapes=True, optimize='dp')[1]
    assert max(info.scale_list) == 3