import numpy as np
import pytest
import cirq
def test_via_unitary():

    class No1:

        def _unitary_(self):
            return NotImplemented

    class No2:

        def _unitary_(self):
            return None

    class Yes:

        def _unitary_(self):
            return np.array([[1]])
    assert not cirq.has_unitary(No1())
    assert not cirq.has_unitary(No2())
    assert cirq.has_unitary(Yes())
    assert cirq.has_unitary(Yes(), allow_decompose=False)