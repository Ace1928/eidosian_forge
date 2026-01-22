from itertools import combinations, product
from random import randint
from string import ascii_lowercase as alphabet
from typing import Optional, Sequence, Tuple
import numpy
import pytest
import cirq
import cirq.testing as ct
import cirq.contrib.acquaintance as cca
def test_swap_network_gate():
    qubits = tuple((cirq.NamedQubit(s) for s in alphabet))
    acquaintance_size = 3
    n_parts = 3
    part_lens = (acquaintance_size - 1,) * n_parts
    n_qubits = sum(part_lens)
    swap_network_op = cca.SwapNetworkGate(part_lens, acquaintance_size=acquaintance_size)(*qubits[:n_qubits])
    swap_network = cirq.Circuit(swap_network_op)
    expected_text_diagram = '\na: ───×(0,0)───\n      │\nb: ───×(0,1)───\n      │\nc: ───×(1,0)───\n      │\nd: ───×(1,1)───\n      │\ne: ───×(2,0)───\n      │\nf: ───×(2,1)───\n    '.strip()
    ct.assert_has_diagram(swap_network, expected_text_diagram)
    no_decomp = lambda op: isinstance(op.gate, (cca.CircularShiftGate, cca.LinearPermutationGate))
    swap_network = cirq.expand_composite(swap_network, no_decomp=no_decomp)
    expected_text_diagram = '\na: ───█───────╲0╱───█─────────────────█───────────╲0╱───█───────0↦1───\n      │       │     │                 │           │     │       │\nb: ───█───█───╲1╱───█───█─────────────█───█───────╲1╱───█───█───1↦0───\n      │   │   │     │   │             │   │       │     │   │   │\nc: ───█───█───╱2╲───█───█───█───╲0╱───█───█───█───╱2╲───█───█───2↦3───\n          │   │         │   │   │         │   │   │         │   │\nd: ───────█───╱3╲───█───█───█───╲1╱───█───█───█───╱3╲───────█───3↦2───\n                    │       │   │     │       │                 │\ne: ─────────────────█───────█───╱2╲───█───────█─────────────────4↦5───\n                    │           │     │                         │\nf: ─────────────────█───────────╱3╲───█─────────────────────────5↦4───\n    '.strip()
    ct.assert_has_diagram(swap_network, expected_text_diagram)
    acquaintance_size = 3
    n_parts = 6
    part_lens = (1,) * n_parts
    n_qubits = sum(part_lens)
    swap_network_op = cca.SwapNetworkGate(part_lens, acquaintance_size=acquaintance_size)(*qubits[:n_qubits])
    swap_network = cirq.Circuit(swap_network_op)
    no_decomp = lambda op: isinstance(op.gate, cca.CircularShiftGate)
    swap_network = cirq.expand_composite(swap_network, no_decomp=no_decomp)
    expected_text_diagram = '\na: ───╲0╱─────────╲0╱─────────╲0╱─────────\n      │           │           │\nb: ───╱1╲───╲0╱───╱1╲───╲0╱───╱1╲───╲0╱───\n            │           │           │\nc: ───╲0╱───╱1╲───╲0╱───╱1╲───╲0╱───╱1╲───\n      │           │           │\nd: ───╱1╲───╲0╱───╱1╲───╲0╱───╱1╲───╲0╱───\n            │           │           │\ne: ───╲0╱───╱1╲───╲0╱───╱1╲───╲0╱───╱1╲───\n      │           │           │\nf: ───╱1╲─────────╱1╲─────────╱1╲─────────\n    '.strip()
    ct.assert_has_diagram(swap_network, expected_text_diagram)