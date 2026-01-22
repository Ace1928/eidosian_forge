import random
import pytest
import cirq
import cirq.testing as ct
import cirq.contrib.acquaintance as cca
def test_display_mapping():
    indices = [4, 2, 0, 1, 3]
    qubits = cirq.LineQubit.range(len(indices))
    circuit = cca.complete_acquaintance_strategy(qubits, 2)
    cca.expose_acquaintance_gates(circuit)
    initial_mapping = dict(zip(qubits, indices))
    cca.display_mapping(circuit, initial_mapping)
    expected_diagram = '\n0: ───4───█───4───╲0╱───2───────2─────────2───█───2───╲0╱───1───────1─────────1───█───1───╲0╱───3───\n          │       │                           │       │                           │       │\n1: ───2───█───2───╱1╲───4───█───4───╲0╱───1───█───1───╱1╲───2───█───2───╲0╱───3───█───3───╱1╲───1───\n                            │       │                           │       │\n2: ───0───█───0───╲0╱───1───█───1───╱1╲───4───█───4───╲0╱───3───█───3───╱1╲───2───█───2───╲0╱───0───\n          │       │                           │       │                           │       │\n3: ───1───█───1───╱1╲───0───█───0───╲0╱───3───█───3───╱1╲───4───█───4───╲0╱───0───█───0───╱1╲───2───\n                            │       │                           │       │\n4: ───3───────3─────────3───█───3───╱1╲───0───────0─────────0───█───0───╱1╲───4───────4─────────4───\n'
    cirq.testing.assert_has_diagram(circuit, expected_diagram)