import itertools
import re
from typing import cast, Tuple, Union
import numpy as np
import pytest
import sympy
import cirq
from cirq import protocols
from cirq.type_workarounds import NotImplementedType
def test_non_diagrammable_subop():
    qbits = cirq.LineQubit.range(2)

    class UndiagrammableGate(cirq.testing.SingleQubitGate):

        def _has_mixture_(self):
            return True
    undiagrammable_op = UndiagrammableGate()(qbits[1])
    c_op = cirq.ControlledOperation(qbits[:1], undiagrammable_op)
    assert cirq.circuit_diagram_info(c_op, default=None) is None