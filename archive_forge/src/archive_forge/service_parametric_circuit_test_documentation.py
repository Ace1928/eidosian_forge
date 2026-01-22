from typing import cast, Tuple
import pytest
import cirq
from pyquil import get_qc
from pyquil.api import QVM
from cirq_rigetti import RigettiQCSService
test that RigettiQCSService can run a basic parametric circuit on
    the QVM and return an accurate `cirq.study.Result`.
    