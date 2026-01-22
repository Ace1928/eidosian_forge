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
@pytest.mark.parametrize('sampler_engine', [cirq.Simulator, _MOCK_ENGINE_SAMPLER])
def test_run_local(sampler_engine, monkeypatch):
    called_times = 0

    def myfunc(calibration: LocalXEBPhasedFSimCalibrationRequest, sampler: cirq.Sampler):
        nonlocal called_times
        assert isinstance(calibration, LocalXEBPhasedFSimCalibrationRequest)
        assert sampler is not None
        called_times += 1
        return []
    monkeypatch.setattr('cirq_google.calibration.workflow.run_local_xeb_calibration', myfunc)
    qubit_indices = [(0, 5), (0, 6), (1, 6), (2, 6)]
    qubits = [cirq.GridQubit(*idx) for idx in qubit_indices]
    circuits = [random_rotations_between_grid_interaction_layers_circuit(qubits, depth=depth, two_qubit_op_factory=lambda a, b, _: SQRT_ISWAP_INV_GATE.on(a, b), pattern=cirq.experiments.GRID_ALIGNED_PATTERN, seed=10) for depth in [5, 10]]
    options = LocalXEBPhasedFSimCalibrationOptions(fsim_options=XEBPhasedFSimCharacterizationOptions(characterize_zeta=True, characterize_gamma=True, characterize_chi=True, characterize_theta=False, characterize_phi=False, theta_default=np.pi / 4), n_processes=1)
    characterization_requests = []
    for circuit in circuits:
        _, characterization_requests = workflow.prepare_characterization_for_moments(circuit, options=options, initial=characterization_requests)
    assert len(characterization_requests) == 2
    for cr in characterization_requests:
        assert isinstance(cr, LocalXEBPhasedFSimCalibrationRequest)
    workflow.run_calibrations(characterization_requests, sampler_engine)
    assert called_times == 2