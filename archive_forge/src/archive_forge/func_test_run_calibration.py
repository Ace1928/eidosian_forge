import datetime
from unittest import mock
import time
import numpy as np
import pytest
import duet
from google.protobuf import any_pb2, timestamp_pb2
from google.protobuf.text_format import Merge
import cirq
import cirq_google
import cirq_google as cg
from cirq_google.api import v1, v2
from cirq_google.engine import util
from cirq_google.cloud import quantum
from cirq_google.engine.engine import EngineContext
@mock.patch('cirq_google.engine.engine_client.EngineClient', autospec=True)
def test_run_calibration(client):
    setup_run_circuit_with_result_(client, _CALIBRATION_RESULTS_V2)
    engine = cg.Engine(project_id='proj', proto_version=cg.engine.engine.ProtoVersion.V2)
    q1 = cirq.GridQubit(2, 3)
    q2 = cirq.GridQubit(2, 4)
    layer1 = cg.CalibrationLayer('xeb', cirq.Circuit(cirq.CZ(q1, q2)), {'num_layers': 42})
    layer2 = cg.CalibrationLayer('readout', cirq.Circuit(cirq.measure(q1, q2)), {'num_samples': 4242})
    job = engine.run_calibration(layers=[layer1, layer2], job_id='job-id', processor_id='mysim')
    results = job.calibration_results()
    assert len(results) == 2
    assert results[0].code == v2.calibration_pb2.SUCCESS
    assert results[0].error_message == 'First success'
    assert results[0].token == 'abc123'
    assert len(results[0].metrics) == 1
    assert len(results[0].metrics['fidelity']) == 1
    assert results[0].metrics['fidelity'][q1, q2] == [0.75]
    assert results[1].code == v2.calibration_pb2.SUCCESS
    assert results[1].error_message == 'Second success'
    client().create_job_async.assert_called_once_with(project_id='proj', program_id='prog', job_id='job-id', processor_ids=['mysim'], run_context=util.pack_any(v2.run_context_pb2.RunContext()), description=None, labels={'calibration': ''})