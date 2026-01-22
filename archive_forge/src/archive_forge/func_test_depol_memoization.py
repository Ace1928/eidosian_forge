from typing import Dict, List, Set, Tuple
import numpy as np
import cirq
import pytest
from cirq.devices.noise_properties import NoiseModelFromNoiseProperties
from cirq.devices.superconducting_qubits_noise_properties import (
from cirq.devices.noise_utils import OpIdentifier, PHYSICAL_GATE_TAG
def test_depol_memoization():
    q0 = cirq.LineQubit(0)
    props = ExampleNoiseProperties(**default_props([q0], []))
    depol_error_a = props._depolarizing_error
    depol_error_b = props._depolarizing_error
    assert depol_error_a == depol_error_b
    assert depol_error_a is depol_error_b