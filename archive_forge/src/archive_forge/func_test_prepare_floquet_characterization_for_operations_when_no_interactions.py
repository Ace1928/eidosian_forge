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
def test_prepare_floquet_characterization_for_operations_when_no_interactions():
    q00 = cirq.GridQubit(0, 0)
    q11 = cirq.GridQubit(1, 1)
    circuit = cirq.Circuit([cirq.X(q00), cirq.X(q11)])
    assert workflow.prepare_floquet_characterization_for_operations(circuit) == []