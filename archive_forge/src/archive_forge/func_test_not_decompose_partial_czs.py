from typing import Optional, Sequence, Type
import pytest
import cirq
import sympy
import numpy as np
def test_not_decompose_partial_czs():
    circuit = cirq.Circuit(cirq.CZPowGate(exponent=0.1, global_shift=-0.5)(*cirq.LineQubit.range(2)))
    cirq.optimize_for_target_gateset(circuit, gateset=cirq.CZTargetGateset(), ignore_failures=False)
    cz_gates = [op.gate for op in circuit.all_operations() if isinstance(op, cirq.GateOperation) and isinstance(op.gate, cirq.CZPowGate)]
    num_full_cz = sum((1 for cz in cz_gates if cz.exponent % 2 == 1))
    num_part_cz = sum((1 for cz in cz_gates if cz.exponent % 2 != 1))
    assert num_full_cz == 0
    assert num_part_cz == 1