import itertools
import pytest
import sympy
import cirq
def test_to_sweeps_invalid():
    with pytest.raises(TypeError, match='Unrecognized sweepable'):
        cirq.study.to_sweeps('nope')