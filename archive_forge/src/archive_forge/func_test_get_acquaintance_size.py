from itertools import combinations, product
from random import randint
from string import ascii_lowercase as alphabet
from typing import Optional, Sequence, Tuple
import numpy
import pytest
import cirq
import cirq.testing as ct
import cirq.contrib.acquaintance as cca
def test_get_acquaintance_size():
    qubits = cirq.LineQubit.range(8)
    op = OtherOperation(qubits)
    assert op.with_qubits(qubits) == op
    assert cca.get_acquaintance_size(op) == 0
    for s, _ in enumerate(qubits):
        op = cca.acquaint(*qubits[:s + 1])
        assert cca.get_acquaintance_size(op) == s + 1
    part_lens = (2, 2, 2, 2)
    acquaintance_size = 3
    gate = cca.SwapNetworkGate(part_lens, acquaintance_size)
    op = gate(*qubits[:sum(part_lens)])
    assert cca.get_acquaintance_size(op) == 3
    part_lens = (2, 2, 2, 2)
    acquaintance_size = 4
    gate = cca.SwapNetworkGate(part_lens, acquaintance_size)
    op = gate(*qubits[:sum(part_lens)])
    assert cca.get_acquaintance_size(op) == 0
    part_lens = (2, 2, 2, 2)
    acquaintance_size = 1
    gate = cca.SwapNetworkGate(part_lens, acquaintance_size)
    op = gate(*qubits[:sum(part_lens)])
    assert cca.get_acquaintance_size(op) == 0
    part_lens = (2, 2, 2, 2)
    acquaintance_size = 1
    gate = cca.SwapNetworkGate(part_lens, acquaintance_size)
    op = gate(*qubits[:sum(part_lens)])
    assert cca.get_acquaintance_size(op) == 0