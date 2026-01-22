from typing import Optional
import cirq
import pytest
import sympy
import numpy as np
def test_optimizes_single_iswap_require0_raises():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.CNOT(a, b))
    with pytest.raises(ValueError, match='cannot be decomposed into exactly 0 sqrt-iSWAP gates'):
        _ = cirq.optimize_for_target_gateset(c, gateset=cirq.SqrtIswapTargetGateset(required_sqrt_iswap_count=0), ignore_failures=False)