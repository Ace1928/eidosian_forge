from typing import List, Tuple
import re
import numpy as np
import pytest
import sympy
import cirq
from cirq import value
from cirq.testing import assert_has_consistent_trace_distance_bound
def test_approx_eq_periodic():
    assert cirq.approx_eq(CExpZinGate(1.5), CExpZinGate(5.5), atol=1e-09)
    assert cirq.approx_eq(CExpZinGate(1.5), CExpZinGate(9.5), atol=1e-09)
    assert cirq.approx_eq(CExpZinGate(-2.5), CExpZinGate(1.5), atol=1e-09)
    assert not cirq.approx_eq(CExpZinGate(0), CExpZinGate(1.5), atol=1e-09)
    assert cirq.approx_eq(CExpZinGate(0 - 1e-10), CExpZinGate(0), atol=1e-09)
    assert cirq.approx_eq(CExpZinGate(0), CExpZinGate(4 - 1e-10), atol=1e-09)