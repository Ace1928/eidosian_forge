from typing import Iterable, List, Sequence, Tuple
import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('decomposed_cls', [HasKraus, HasMixture, HasUnitary])
def test_has_kraus_when_decomposed(decomposed_cls):
    op = HasKrausWhenDecomposed(decomposed_cls).on(cirq.NamedQubit('test'))
    assert cirq.has_kraus(op)
    assert not cirq.has_kraus(op, allow_decompose=False)