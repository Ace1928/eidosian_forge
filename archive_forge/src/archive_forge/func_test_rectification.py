from string import ascii_lowercase as alphabet
import pytest
import cirq
import cirq.contrib.acquaintance as cca
def test_rectification():
    qubits = cirq.LineQubit.range(4)
    perm_gate = cca.SwapPermutationGate()
    operations = [perm_gate(*qubits[:2]), cca.acquaint(*qubits[2:]), cca.acquaint(*qubits[:2]), perm_gate(*qubits[2:])]
    strategy = cirq.Circuit(operations)
    cca.rectify_acquaintance_strategy(strategy)
    actual_text_diagram = strategy.to_text_diagram().strip()
    expected_text_diagram = '\n0: ───────0↦1─────────█───\n          │           │\n1: ───────1↦0─────────█───\n\n2: ───█─────────0↦1───────\n      │         │\n3: ───█─────────1↦0───────\n    '.strip()
    assert actual_text_diagram == expected_text_diagram
    strategy = cirq.Circuit(operations)
    cca.rectify_acquaintance_strategy(strategy, False)
    actual_text_diagram = strategy.to_text_diagram()
    expected_text_diagram = '\n0: ───0↦1───────█─────────\n      │         │\n1: ───1↦0───────█─────────\n\n2: ─────────█───────0↦1───\n            │       │\n3: ─────────█───────1↦0───\n    '.strip()
    assert actual_text_diagram == expected_text_diagram