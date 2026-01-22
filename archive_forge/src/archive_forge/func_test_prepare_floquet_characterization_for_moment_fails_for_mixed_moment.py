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
def test_prepare_floquet_characterization_for_moment_fails_for_mixed_moment():
    a, b, c = cirq.LineQubit.range(3)
    moment = cirq.Moment([cirq.FSimGate(theta=np.pi / 4, phi=0.0).on(a, b), cirq.Z.on(c)])
    with pytest.raises(workflow.IncompatibleMomentError):
        workflow.prepare_floquet_characterization_for_moment(moment, WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION)