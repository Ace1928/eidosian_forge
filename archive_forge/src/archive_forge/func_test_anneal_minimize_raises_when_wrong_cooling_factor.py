import math
from unittest import mock
import pytest
from cirq_google.line.placement import optimization
def test_anneal_minimize_raises_when_wrong_cooling_factor():
    with pytest.raises(ValueError):
        optimization.anneal_minimize('initial', lambda s: 1.0 if s == 'initial' else 0.0, lambda s: 'better', lambda: 1.0, 1.0, 0.5, 2.0, 1)