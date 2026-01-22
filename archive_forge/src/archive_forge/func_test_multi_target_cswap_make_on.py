import random
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_multi_target_cswap_make_on():
    qubits = cirq.LineQubit.range(5)
    c, q_x, q_y = (qubits[:1], qubits[1:3], qubits[3:])
    cswap1 = cirq_ft.MultiTargetCSwap(2).on_registers(control=c, target_x=q_x, target_y=q_y)
    cswap2 = cirq_ft.MultiTargetCSwap.make_on(control=c, target_x=q_x, target_y=q_y)
    assert cswap1 == cswap2