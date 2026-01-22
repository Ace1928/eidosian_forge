from typing import cast, Tuple
import pytest
import cirq
from pyquil import get_qc
from pyquil.api import QVM
from cirq_rigetti import RigettiQCSSampler, circuit_sweep_executors
test that RigettiQCSSampler can run a basic parametric circuit on the QVM using parametric
    compilation and return an accurate list of `cirq.study.Result`.
    