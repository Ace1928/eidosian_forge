import functools
import itertools
import math
import random
import numpy as np
import pytest
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.ops.boolean_hamiltonian as bh
@pytest.mark.parametrize('input_cnots,input_flip_control_and_target,expected_simplified,expected_output_cnots', [([], False, False, []), ([], True, False, []), ([(0, 1)], False, False, [(0, 1)]), ([(0, 1)], True, False, [(0, 1)]), ([(0, 1), (0, 1)], False, True, []), ([(0, 1), (0, 1)], True, True, []), ([(0, 1), (2, 1), (0, 1)], False, True, [(2, 1)]), ([(0, 1), (0, 2), (0, 1)], True, True, [(0, 2)]), ([(0, 1), (0, 2), (0, 1)], False, False, [(0, 1), (0, 2), (0, 1)]), ([(0, 1), (2, 1), (0, 1)], True, False, [(0, 1), (2, 1), (0, 1)]), ([(0, 1), (2, 3), (0, 1)], False, False, [(0, 1), (2, 3), (0, 1)]), ([(0, 1), (2, 3), (2, 3), (0, 1)], False, True, []), ([(0, 1), (2, 1), (2, 3), (2, 3), (0, 1)], False, True, [(2, 1)])])
def test_simplify_commuting_cnots(input_cnots, input_flip_control_and_target, expected_simplified, expected_output_cnots):
    actual_simplified, actual_output_cnots = bh._simplify_commuting_cnots(input_cnots, input_flip_control_and_target)
    assert actual_simplified == expected_simplified
    assert actual_output_cnots == expected_output_cnots