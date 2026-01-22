from typing import Iterable, List, Sequence, Tuple
import numpy as np
import pytest
import cirq
def test_kraus_no_methods():

    class NoMethod:
        pass
    with pytest.raises(TypeError, match='no _kraus_ or _mixture_ or _unitary_ method'):
        _ = cirq.kraus(NoMethod())
    assert cirq.kraus(NoMethod(), None) is None
    assert cirq.kraus(NoMethod, NotImplemented) is NotImplemented
    assert cirq.kraus(NoMethod(), (1,)) == (1,)
    assert cirq.kraus(NoMethod(), LOCAL_DEFAULT) is LOCAL_DEFAULT
    assert not cirq.has_kraus(NoMethod())