import os
from typing import cast
from unittest import mock
import numpy as np
import pandas as pd
import pytest
import sympy
from google.protobuf import text_format
import cirq
import cirq_google
from cirq.experiments.xeb_fitting import XEBPhasedFSimCharacterizationOptions
from cirq_google.api import v2
from cirq_google.calibration.phased_fsim import (
from cirq_google.serialization.arg_func_langs import arg_to_proto
def test_try_convert_sqrt_iswap_to_fsim_converts_correctly():
    expected = cirq.FSimGate(theta=np.pi / 4, phi=0)
    expected_unitary = cirq.unitary(expected)
    fsim = cirq.FSimGate(theta=np.pi / 4, phi=0)
    assert np.allclose(cirq.unitary(fsim), expected_unitary)
    assert try_convert_sqrt_iswap_to_fsim(fsim) == PhaseCalibratedFSimGate(expected, 0.0)
    assert try_convert_sqrt_iswap_to_fsim(cirq.FSimGate(theta=np.pi / 4, phi=0.1)) is None
    assert try_convert_sqrt_iswap_to_fsim(cirq.FSimGate(theta=np.pi / 3, phi=0)) is None
    phased_fsim = cirq.PhasedFSimGate(theta=np.pi / 4, phi=0)
    assert np.allclose(cirq.unitary(phased_fsim), expected_unitary)
    assert try_convert_sqrt_iswap_to_fsim(phased_fsim) == PhaseCalibratedFSimGate(expected, 0.0)
    assert try_convert_sqrt_iswap_to_fsim(cirq.PhasedFSimGate(theta=np.pi / 4, zeta=0.1, phi=0)) is None
    iswap_pow = cirq.ISwapPowGate(exponent=-0.5)
    assert np.allclose(cirq.unitary(iswap_pow), expected_unitary)
    assert try_convert_sqrt_iswap_to_fsim(iswap_pow) == PhaseCalibratedFSimGate(expected, 0.0)
    assert try_convert_sqrt_iswap_to_fsim(cirq.ISwapPowGate(exponent=-0.4)) is None
    phased_iswap_pow = cirq.PhasedISwapPowGate(exponent=0.5, phase_exponent=-0.5)
    assert np.allclose(cirq.unitary(phased_iswap_pow), expected_unitary)
    assert try_convert_sqrt_iswap_to_fsim(phased_iswap_pow) == PhaseCalibratedFSimGate(expected, 0.0)
    assert try_convert_sqrt_iswap_to_fsim(cirq.PhasedISwapPowGate(exponent=-0.6, phase_exponent=0.1)) is None
    assert try_convert_sqrt_iswap_to_fsim(cirq.CZ) is None
    assert try_convert_sqrt_iswap_to_fsim(cirq.FSimGate(theta=sympy.Symbol('t'), phi=sympy.Symbol('p'))) is None