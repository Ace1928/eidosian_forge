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
def test_run_calibrations_fails_when_invalid_arguments():
    with pytest.raises(ValueError):
        assert workflow.run_calibrations([], None, 'qproc', max_layers_per_request=0)
    request = FloquetPhasedFSimCalibrationRequest(gate=SQRT_ISWAP_INV_GATE, pairs=(), options=WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION)
    engine = mock.MagicMock(spec=cirq_google.Engine)
    with pytest.raises(ValueError):
        assert workflow.run_calibrations([request], engine, None)
    with pytest.raises(ValueError):
        assert workflow.run_calibrations([request], 0, 'qproc')