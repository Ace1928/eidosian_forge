import itertools
import sys
import numpy as np
import pytest
import opt_einsum as oe
def test_custom_dp_can_optimize_for_outer_products():
    eq = 'a,b,abc->c'
    da, db, dc = (2, 2, 3)
    shapes = [(da,), (db,), (da, db, dc)]
    opt1 = oe.DynamicProgramming(search_outer=False)
    opt2 = oe.DynamicProgramming(search_outer=True)
    info1 = oe.contract_path(eq, *shapes, shapes=True, optimize=opt1)[1]
    info2 = oe.contract_path(eq, *shapes, shapes=True, optimize=opt2)[1]
    assert info2.opt_cost < info1.opt_cost