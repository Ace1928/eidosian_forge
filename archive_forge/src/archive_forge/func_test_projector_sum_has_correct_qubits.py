import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
@pytest.mark.parametrize('terms, expected_qubits', (([], ()), ([cirq.ProjectorString({q0: 0})], (q0,)), ([cirq.ProjectorString({q0: 0}), cirq.ProjectorString({q1: 0})], (q0, q1)), ([cirq.ProjectorString({q0: 0, q1: 1})], (q0, q1))))
def test_projector_sum_has_correct_qubits(terms, expected_qubits):
    combination = cirq.ProjectorSum.from_projector_strings(terms)
    assert combination.qubits == expected_qubits