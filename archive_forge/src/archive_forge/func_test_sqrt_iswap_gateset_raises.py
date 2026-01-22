from typing import Optional
import cirq
import pytest
import sympy
import numpy as np
def test_sqrt_iswap_gateset_raises():
    with pytest.raises(ValueError, match='`required_sqrt_iswap_count` must be 0, 1, 2, or 3'):
        _ = cirq.SqrtIswapTargetGateset(required_sqrt_iswap_count=4)