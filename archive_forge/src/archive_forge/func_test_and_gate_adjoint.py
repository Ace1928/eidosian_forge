import itertools
import random
from typing import List, Tuple
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@pytest.mark.parametrize('cv', [(0, 0), (0, 1), (1, 0), (1, 1)])
@allow_deprecated_cirq_ft_use_in_tests
def test_and_gate_adjoint(cv: Tuple[int, int]):
    c1, c2, t = cirq.LineQubit.range(3)
    all_cvs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    input_states = [inp + (1 if inp == cv else 0,) for inp in all_cvs]
    output_states = [inp + (0,) for inp in all_cvs]
    circuit = cirq.Circuit(cirq_ft.And(cv, adjoint=True).on(c1, c2, t))
    for inp, out in zip(input_states, output_states):
        cirq_ft.testing.assert_circuit_inp_out_cirqsim(circuit, [c1, c2, t], inp, out)