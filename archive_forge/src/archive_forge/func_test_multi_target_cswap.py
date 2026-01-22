import random
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_multi_target_cswap():
    qubits = cirq.LineQubit.range(5)
    c, q_x, q_y = (qubits[0], qubits[1:3], qubits[3:])
    cswap = cirq_ft.MultiTargetCSwap(2).on_registers(control=c, target_x=q_x, target_y=q_y)
    cswap_approx = cirq_ft.MultiTargetCSwapApprox(2).on_registers(control=c, target_x=q_x, target_y=q_y)
    setup_code = 'import cirq\nimport cirq_ft'
    cirq.testing.assert_implements_consistent_protocols(cswap, setup_code=setup_code)
    cirq.testing.assert_implements_consistent_protocols(cswap_approx, setup_code=setup_code)
    circuit = cirq.Circuit(cswap, cswap_approx)
    cirq.testing.assert_has_diagram(circuit, '\n0: ───@──────@(approx)───\n      │      │\n1: ───×(x)───×(x)────────\n      │      │\n2: ───×(x)───×(x)────────\n      │      │\n3: ───×(y)───×(y)────────\n      │      │\n4: ───×(y)───×(y)────────\n    ')
    cirq.testing.assert_has_diagram(circuit, '\n0: ---@--------@(approx)---\n      |        |\n1: ---swap_x---swap_x------\n      |        |\n2: ---swap_x---swap_x------\n      |        |\n3: ---swap_y---swap_y------\n      |        |\n4: ---swap_y---swap_y------\n    ', use_unicode_characters=False)