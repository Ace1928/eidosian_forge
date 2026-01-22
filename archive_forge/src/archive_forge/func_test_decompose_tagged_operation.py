import itertools
from typing import Optional
from unittest import mock
import pytest
import cirq
def test_decompose_tagged_operation():
    op = cirq.TaggedOperation(cirq.CircuitOperation(circuit=cirq.FrozenCircuit([cirq.Moment(cirq.SWAP(cirq.LineQubit(0), cirq.LineQubit(1)))])), 'tag')
    assert cirq.decompose_once(op) == cirq.decompose_once(op.untagged)