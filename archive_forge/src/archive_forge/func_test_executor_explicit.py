from itertools import combinations
from string import ascii_lowercase
from typing import Sequence, Dict, Tuple
import numpy as np
import pytest
import cirq
import cirq.testing as ct
import cirq.contrib.acquaintance as cca
def test_executor_explicit():
    num_qubits = 8
    qubits = cirq.LineQubit.range(num_qubits)
    circuit = cca.complete_acquaintance_strategy(qubits, 2)
    gates = {(i, j): ExampleGate([str(k) for k in ij]) for ij in combinations(range(num_qubits), 2) for i, j in (ij, ij[::-1])}
    initial_mapping = {q: i for i, q in enumerate(sorted(qubits))}
    execution_strategy = cca.GreedyExecutionStrategy(gates, initial_mapping)
    with pytest.raises(ValueError):
        executor = cca.StrategyExecutorTransformer(None)
    executor = cca.StrategyExecutorTransformer(execution_strategy)
    with pytest.raises(NotImplementedError):
        bad_gates = {(0,): ExampleGate(['0']), (0, 1): ExampleGate(['0', '1'])}
        cca.GreedyExecutionStrategy(bad_gates, initial_mapping)
    with pytest.raises(TypeError):
        bad_strategy = cirq.Circuit(cirq.X(qubits[0]))
        executor(bad_strategy)
    circuit = executor(circuit)
    expected_text_diagram = '\n0: ───0───1───╲0╱─────────────────1───3───╲0╱─────────────────3───5───╲0╱─────────────────5───7───╲0╱─────────────────\n      │   │   │                   │   │   │                   │   │   │                   │   │   │\n1: ───1───0───╱1╲───0───3───╲0╱───3───1───╱1╲───1───5───╲0╱───5───3───╱1╲───3───7───╲0╱───7───5───╱1╲───5───6───╲0╱───\n                    │   │   │                   │   │   │                   │   │   │                   │   │   │\n2: ───2───3───╲0╱───3───0───╱1╲───0───5───╲0╱───5───1───╱1╲───1───7───╲0╱───7───3───╱1╲───3───6───╲0╱───6───5───╱1╲───\n      │   │   │                   │   │   │                   │   │   │                   │   │   │\n3: ───3───2───╱1╲───2───5───╲0╱───5───0───╱1╲───0───7───╲0╱───7───1───╱1╲───1───6───╲0╱───6───3───╱1╲───3───4───╲0╱───\n                    │   │   │                   │   │   │                   │   │   │                   │   │   │\n4: ───4───5───╲0╱───5───2───╱1╲───2───7───╲0╱───7───0───╱1╲───0───6───╲0╱───6───1───╱1╲───1───4───╲0╱───4───3───╱1╲───\n      │   │   │                   │   │   │                   │   │   │                   │   │   │\n5: ───5───4───╱1╲───4───7───╲0╱───7───2───╱1╲───2───6───╲0╱───6───0───╱1╲───0───4───╲0╱───4───1───╱1╲───1───2───╲0╱───\n                    │   │   │                   │   │   │                   │   │   │                   │   │   │\n6: ───6───7───╲0╱───7───4───╱1╲───4───6───╲0╱───6───2───╱1╲───2───4───╲0╱───4───0───╱1╲───0───2───╲0╱───2───1───╱1╲───\n      │   │   │                   │   │   │                   │   │   │                   │   │   │\n7: ───7───6───╱1╲─────────────────6───4───╱1╲─────────────────4───2───╱1╲─────────────────2───0───╱1╲─────────────────\n    '.strip()
    ct.assert_has_diagram(circuit, expected_text_diagram)