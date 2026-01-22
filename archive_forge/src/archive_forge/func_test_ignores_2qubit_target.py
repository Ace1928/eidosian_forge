from typing import List
import numpy as np
import pytest
import cirq
def test_ignores_2qubit_target():
    c = cirq.Circuit(cirq.CZ(*cirq.LineQubit.range(2)))
    assert_optimizes(optimized=cirq.merge_k_qubit_unitaries(c, k=1), expected=c)