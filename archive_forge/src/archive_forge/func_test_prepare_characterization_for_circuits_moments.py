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
def test_prepare_characterization_for_circuits_moments():
    a, b, c, d = cirq.LineQubit.range(4)
    circuit_1 = cirq.Circuit([[cirq.X(a), cirq.Y(c)], [SQRT_ISWAP_INV_GATE.on(a, b), SQRT_ISWAP_INV_GATE.on(c, d)], [cirq.WaitGate(duration=cirq.Duration(micros=5.0)).on(b)]])
    circuit_2 = cirq.Circuit([[cirq.X(a), cirq.Y(c)], [SQRT_ISWAP_INV_GATE.on(b, c)], [cirq.WaitGate(duration=cirq.Duration(micros=5.0)).on(b)]])
    options = WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION
    circuits_with_calibration, requests = workflow.prepare_characterization_for_circuits_moments([circuit_1, circuit_2], options=options)
    assert requests == [cirq_google.calibration.FloquetPhasedFSimCalibrationRequest(pairs=((a, b), (c, d)), gate=SQRT_ISWAP_INV_GATE, options=options), cirq_google.calibration.FloquetPhasedFSimCalibrationRequest(pairs=((b, c),), gate=SQRT_ISWAP_INV_GATE, options=options)]
    assert len(circuits_with_calibration) == 2
    assert circuits_with_calibration[0].circuit == circuit_1
    assert circuits_with_calibration[0].moment_to_calibration == [None, 0, None]
    assert circuits_with_calibration[1].circuit == circuit_2
    assert circuits_with_calibration[1].moment_to_calibration == [None, 1, None]