import itertools
import re
from typing import cast, Tuple, Union
import numpy as np
import pytest
import sympy
import cirq
from cirq import protocols
from cirq.type_workarounds import NotImplementedType
def test_controlled_operation_gate():
    gate = cirq.X.controlled(control_values=[0, 1], control_qid_shape=[2, 3])
    op = gate.on(cirq.LineQubit(0), cirq.LineQid(1, 3), cirq.LineQubit(2))
    assert op.gate == gate

    class Gateless(cirq.Operation):

        @property
        def qubits(self):
            return ()

        def with_qubits(self, *new_qubits):
            return self

        def _has_mixture_(self):
            return True
    op = Gateless().controlled_by(cirq.LineQubit(0))
    assert op.gate is None