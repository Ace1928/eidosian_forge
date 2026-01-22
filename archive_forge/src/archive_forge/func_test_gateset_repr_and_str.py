from typing import Tuple, List, cast
import re
import pytest
import sympy
import cirq
from cirq._compat import proper_repr
import numpy as np
@pytest.mark.parametrize('g', [gateset, cirq.Gateset(name='empty gateset')])
def test_gateset_repr_and_str(g):
    cirq.testing.assert_equivalent_repr(g)
    assert g.name in str(g)
    for gate_family in g.gates:
        assert str(gate_family) in str(g)