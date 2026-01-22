from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
@pytest.mark.parametrize('op', [cirq.X(cirq.NamedQubit('q')), cirq.X(cirq.NamedQubit('q')).with_tags('tagged_op')])
def test_with_methods_return_self_on_empty_conditions(op):
    assert op is op.with_tags(*[])
    assert op is op.with_classical_controls(*[])
    assert op is op.controlled_by(*[])