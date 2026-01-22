from typing import Iterable, List, Sequence, Tuple
import numpy as np
import pytest
import cirq
def test_explicit_kraus():
    a0 = np.array([[0, 0], [1, 0]])
    a1 = np.array([[1, 0], [0, 0]])
    c = (a0, a1)

    class ReturnsKraus:

        def _kraus_(self) -> Sequence[np.ndarray]:
            return c
    assert cirq.kraus(ReturnsKraus()) is c
    assert cirq.kraus(ReturnsKraus(), None) is c
    assert cirq.kraus(ReturnsKraus(), NotImplemented) is c
    assert cirq.kraus(ReturnsKraus(), (1,)) is c
    assert cirq.kraus(ReturnsKraus(), LOCAL_DEFAULT) is c
    assert cirq.has_kraus(ReturnsKraus())