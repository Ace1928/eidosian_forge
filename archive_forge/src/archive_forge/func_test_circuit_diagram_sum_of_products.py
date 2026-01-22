from typing import Union, Tuple, cast
import numpy as np
import pytest
import sympy
import cirq
from cirq.type_workarounds import NotImplementedType
def test_circuit_diagram_sum_of_products():
    q = cirq.LineQubit.range(4)
    c = cirq.Circuit(C_xorH.on(*q[:3]), C_01_10_11H.on(*q[:3]), C0C_xorH.on(*q))
    cirq.testing.assert_has_diagram(c, '\n0: ───@────────@(011)───@(00)───\n      │        │        │\n1: ───@(xor)───@(101)───@(01)───\n      │        │        │\n2: ───H────────H────────@(10)───\n                        │\n3: ─────────────────────H───────\n')
    q = cirq.LineQid.for_qid_shape((2, 3, 2))
    c = cirq.Circuit(C_02_20H(*q))
    cirq.testing.assert_has_diagram(c, '\n0 (d=2): ───@(01)───\n            │\n1 (d=3): ───@(20)───\n            │\n2 (d=2): ───H───────\n')