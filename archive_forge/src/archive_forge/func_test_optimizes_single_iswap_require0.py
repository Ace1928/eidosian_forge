from typing import Optional
import cirq
import pytest
import sympy
import numpy as np
def test_optimizes_single_iswap_require0():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.CNOT(a, b), cirq.CNOT(a, b))
    assert_optimization_not_broken(c, required_sqrt_iswap_count=0)
    c = cirq.optimize_for_target_gateset(c, gateset=cirq.SqrtIswapTargetGateset(required_sqrt_iswap_count=0), ignore_failures=False)
    assert len([1 for op in c.all_operations() if len(op.qubits) == 2]) == 0