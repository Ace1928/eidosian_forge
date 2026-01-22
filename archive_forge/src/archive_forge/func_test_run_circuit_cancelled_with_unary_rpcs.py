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
def test_run_circuit_cancelled_with_unary_rpcs(client):
    client().create_program_async.return_value = ('prog', quantum.QuantumProgram(name='projects/proj/programs/prog'))
    client().create_job_async.return_value = ('job-id', quantum.QuantumJob(name='projects/proj/programs/prog/jobs/job-id', execution_status={'state': 'READY'}))
    client().get_job_async.return_value = quantum.QuantumJob(name='projects/proj/programs/prog/jobs/job-id', execution_status={'state': 'CANCELLED'})
    engine = cg.Engine(project_id='proj', context=EngineContext(enable_streaming=False))
    with pytest.raises(RuntimeError, match='Job projects/proj/programs/prog/jobs/job-id failed in state CANCELLED.'):
        engine.run(program=_CIRCUIT)