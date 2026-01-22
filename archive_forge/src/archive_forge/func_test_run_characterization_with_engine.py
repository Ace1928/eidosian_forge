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
def test_run_characterization_with_engine():
    q_00, q_01, q_02, q_03 = [cirq.GridQubit(0, index) for index in range(4)]
    gate = cirq.FSimGate(theta=np.pi / 4, phi=0.0)
    request = FloquetPhasedFSimCalibrationRequest(gate=gate, pairs=((q_00, q_01), (q_02, q_03)), options=FloquetPhasedFSimCalibrationOptions(characterize_theta=True, characterize_zeta=True, characterize_chi=False, characterize_gamma=False, characterize_phi=True))
    result = cirq_google.CalibrationResult(code=cirq_google.api.v2.calibration_pb2.SUCCESS, error_message=None, token=None, valid_until=None, metrics=cirq_google.Calibration(cirq_google.api.v2.metrics_pb2.MetricsSnapshot(metrics=[cirq_google.api.v2.metrics_pb2.Metric(name='angles', targets=['0_qubit_a', '0_qubit_b', '0_theta_est', '0_zeta_est', '0_phi_est', '1_qubit_a', '1_qubit_b', '1_theta_est', '1_zeta_est', '1_phi_est'], values=[cirq_google.api.v2.metrics_pb2.Value(str_val='0_0'), cirq_google.api.v2.metrics_pb2.Value(str_val='0_1'), cirq_google.api.v2.metrics_pb2.Value(double_val=0.1), cirq_google.api.v2.metrics_pb2.Value(double_val=0.2), cirq_google.api.v2.metrics_pb2.Value(double_val=0.3), cirq_google.api.v2.metrics_pb2.Value(str_val='0_2'), cirq_google.api.v2.metrics_pb2.Value(str_val='0_3'), cirq_google.api.v2.metrics_pb2.Value(double_val=0.4), cirq_google.api.v2.metrics_pb2.Value(double_val=0.5), cirq_google.api.v2.metrics_pb2.Value(double_val=0.6)])])))
    job = cirq_google.engine.EngineJob('project_id', 'program_id', 'job_id', None)
    job._calibration_results = [result]
    processor = mock.MagicMock(spec=cirq_google.engine.SimulatedLocalProcessor)
    processor.processor_id = 'qproc'
    engine = cirq_google.engine.SimulatedLocalEngine([processor])
    processor.engine.return_value = engine
    processor.run_calibration.return_value = job
    progress_calls = []

    def progress(step: int, steps: int) -> None:
        progress_calls.append((step, steps))
    actual = workflow.run_calibrations([request], engine, 'qproc', progress_func=progress)
    expected = [PhasedFSimCalibrationResult(parameters={(q_00, q_01): PhasedFSimCharacterization(theta=0.1, zeta=0.2, chi=None, gamma=None, phi=0.3), (q_02, q_03): PhasedFSimCharacterization(theta=0.4, zeta=0.5, chi=None, gamma=None, phi=0.6)}, gate=gate, options=FloquetPhasedFSimCalibrationOptions(characterize_theta=True, characterize_zeta=True, characterize_chi=False, characterize_gamma=False, characterize_phi=True), project_id='project_id', program_id='program_id', job_id='job_id')]
    assert actual == expected
    assert progress_calls == [(1, 1)]