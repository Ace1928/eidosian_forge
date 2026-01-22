from typing import List, Sequence
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
Get LCU coefficients for a 1d ising Hamiltonian.

    The order of the terms is according to `get_1d_Ising_hamiltonian`, namely: ZZ's and X's
    interleaved.
    