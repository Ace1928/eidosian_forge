import dataclasses
import cirq
import cirq_google
import pytest
from cirq_google import (
def test_quantum_executable_group_serialization(tmpdir):
    exes = _get_quantum_executables()
    eg = QuantumExecutableGroup(exes)
    cirq.testing.assert_equivalent_repr(eg, global_vals={'cirq_google': cirq_google})
    cirq.to_json(eg, f'{tmpdir}/eg.json')
    eg_reconstructed = cirq.read_json(f'{tmpdir}/eg.json')
    assert eg == eg_reconstructed