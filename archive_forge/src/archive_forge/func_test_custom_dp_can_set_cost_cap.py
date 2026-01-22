import itertools
import sys
import numpy as np
import pytest
import opt_einsum as oe
def test_custom_dp_can_set_cost_cap():
    eq, shapes = oe.helpers.rand_equation(5, 3, seed=42)
    opt1 = oe.DynamicProgramming(cost_cap=True)
    opt2 = oe.DynamicProgramming(cost_cap=False)
    opt3 = oe.DynamicProgramming(cost_cap=100)
    info1 = oe.contract_path(eq, *shapes, shapes=True, optimize=opt1)[1]
    info2 = oe.contract_path(eq, *shapes, shapes=True, optimize=opt2)[1]
    info3 = oe.contract_path(eq, *shapes, shapes=True, optimize=opt3)[1]
    assert info1.opt_cost == info2.opt_cost == info3.opt_cost