import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_qroam_diagram():
    data = [[1, 2, 3], [4, 5, 6]]
    blocksize = 2
    qrom = cirq_ft.SelectSwapQROM(*data, block_size=blocksize)
    q = cirq.LineQubit.range(cirq.num_qubits(qrom))
    circuit = cirq.Circuit(qrom.on_registers(**infra.split_qubits(qrom.signature, q)))
    cirq.testing.assert_has_diagram(circuit, '\n0: ───In_q──────\n      │\n1: ───In_r──────\n      │\n2: ───QROAM_0───\n      │\n3: ───QROAM_0───\n      │\n4: ───QROAM_1───\n      │\n5: ───QROAM_1───\n      │\n6: ───QROAM_1───\n')