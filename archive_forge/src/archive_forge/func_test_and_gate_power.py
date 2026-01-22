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
@allow_deprecated_cirq_ft_use_in_tests
def test_and_gate_power():
    cv = (1, 0)
    and_gate = cirq_ft.And(cv)
    assert and_gate ** 1 is and_gate
    assert and_gate ** (-1) == cirq_ft.And(cv, adjoint=True)
    assert (and_gate ** (-1)) ** (-1) == cirq_ft.And(cv)