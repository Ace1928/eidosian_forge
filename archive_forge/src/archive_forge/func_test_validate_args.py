from typing import Union, Tuple, cast
import numpy as np
import pytest
import sympy
import cirq
from cirq.type_workarounds import NotImplementedType
def test_validate_args():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    with pytest.raises(ValueError):
        CRestricted.validate_args([])
    with pytest.raises(ValueError):
        CRestricted.validate_args([a])
    CRestricted.validate_args([a, b])
    with pytest.raises(ValueError):
        CY.validate_args([a])
    with pytest.raises(ValueError):
        CY.validate_args([a, b, c])
    CY.validate_args([a, b])
    with pytest.raises(ValueError):
        _ = CY.on()
    with pytest.raises(ValueError):
        _ = CY.on(a)
    with pytest.raises(ValueError):
        _ = CY.on(a, b, c)
    _ = CY.on(a, b)
    with pytest.raises(ValueError):
        _ = CCH.on()
    with pytest.raises(ValueError):
        _ = CCH.on(a)
    with pytest.raises(ValueError):
        _ = CCH.on(a, b)
    with pytest.raises(ValueError, match='Wrong shape of qids'):
        _ = CY.on(q3, b)
    with pytest.raises(ValueError, match='Wrong shape of qids'):
        _ = C2Y.on(a, b)
    with pytest.raises(ValueError, match='Wrong shape of qids'):
        _ = C2C2H.on(a, b, c)
    _ = C2C2H.on(q3, p3, a)