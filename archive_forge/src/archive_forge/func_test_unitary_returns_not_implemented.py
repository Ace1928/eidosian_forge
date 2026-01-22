from typing import Iterable, List, Sequence, Tuple
import numpy as np
import pytest
import cirq
def test_unitary_returns_not_implemented():

    class ReturnsNotImplemented:

        def _unitary_(self):
            return NotImplemented
    with pytest.raises(TypeError, match='returned NotImplemented'):
        _ = cirq.kraus(ReturnsNotImplemented())
    assert cirq.kraus(ReturnsNotImplemented(), None) is None
    assert cirq.kraus(ReturnsNotImplemented(), NotImplemented) is NotImplemented
    assert cirq.kraus(ReturnsNotImplemented(), (1,)) == (1,)
    assert cirq.kraus(ReturnsNotImplemented(), LOCAL_DEFAULT) is LOCAL_DEFAULT