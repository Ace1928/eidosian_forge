import pytest
import sympy
import cirq
def test_list_sweep_str():
    assert str(cirq.UnitSweep) == 'Sweep:\n{}'
    assert str(cirq.Linspace('a', start=0, stop=3, length=4)) == "Sweep:\n{'a': 0.0}\n{'a': 1.0}\n{'a': 2.0}\n{'a': 3.0}"
    assert str(cirq.Linspace('a', start=0, stop=15.75, length=64)) == "Sweep:\n{'a': 0.0}\n{'a': 0.25}\n{'a': 0.5}\n{'a': 0.75}\n{'a': 1.0}\n...\n{'a': 14.75}\n{'a': 15.0}\n{'a': 15.25}\n{'a': 15.5}\n{'a': 15.75}"
    assert str(cirq.ListSweep(cirq.Linspace('a', 0, 3, 4) + cirq.Linspace('b', 1, 2, 2))) == "Sweep:\n{'a': 0.0, 'b': 1.0}\n{'a': 1.0, 'b': 2.0}"
    assert str(cirq.ListSweep(cirq.Linspace('a', 0, 3, 4) * cirq.Linspace('b', 1, 2, 2))) == "Sweep:\n{'a': 0.0, 'b': 1.0}\n{'a': 0.0, 'b': 2.0}\n{'a': 1.0, 'b': 1.0}\n{'a': 1.0, 'b': 2.0}\n{'a': 2.0, 'b': 1.0}\n{'a': 2.0, 'b': 2.0}\n{'a': 3.0, 'b': 1.0}\n{'a': 3.0, 'b': 2.0}"