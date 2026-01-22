from typing import cast, Tuple, List
import cirq
import pytest
import sympy
from pyquil import get_qc
from pyquil.api import QVM
from cirq_rigetti import RigettiQCSSampler
from cirq_rigetti import circuit_transformers
test that RigettiQCSSampler can properly readout qubits after quilc has
    reassigned those qubits in the compiled native Quil.
    