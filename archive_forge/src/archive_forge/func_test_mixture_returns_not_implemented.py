from typing import Iterable, List, Sequence, Tuple
import numpy as np
import pytest
import cirq
def test_mixture_returns_not_implemented():

    class ReturnsNotImplemented:

        def _mixture_(self):
            return NotImplemented
    assert_not_implemented(ReturnsNotImplemented())