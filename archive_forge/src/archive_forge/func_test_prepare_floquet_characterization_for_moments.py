import itertools
from typing import Optional
from unittest import mock
import numpy as np
import pytest
import cirq
import cirq_google
import cirq_google.calibration.workflow as workflow
import cirq_google.calibration.xeb_wrapper
from cirq.experiments import (
from cirq_google.calibration.engine_simulator import PhasedFSimEngineSimulator
from cirq_google.calibration.phased_fsim import (
def test_prepare_floquet_characterization_for_moments():
    a, b, c, d = cirq.LineQubit.range(4)
    circuit = cirq.Circuit([[cirq.X(a), cirq.Y(c)], [SQRT_ISWAP_INV_GATE.on(a, b), SQRT_ISWAP_INV_GATE.on(c, d)], [SQRT_ISWAP_INV_GATE.on(b, c)], [cirq.WaitGate(duration=cirq.Duration(micros=5.0)).on(b)]])
    options = WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION
    circuit_with_calibration, requests = workflow.prepare_floquet_characterization_for_moments(circuit, options=options)
    assert requests == [cirq_google.calibration.FloquetPhasedFSimCalibrationRequest(pairs=((a, b), (c, d)), gate=SQRT_ISWAP_INV_GATE, options=options), cirq_google.calibration.FloquetPhasedFSimCalibrationRequest(pairs=((b, c),), gate=SQRT_ISWAP_INV_GATE, options=options)]
    assert circuit_with_calibration.circuit == circuit
    assert circuit_with_calibration.moment_to_calibration == [None, 0, 1, None]