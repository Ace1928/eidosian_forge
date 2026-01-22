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
def test_bad_result_proto(client):
    result = any_pb2.Any()
    result.CopyFrom(_RESULTS_V2)
    result.type_url = 'type.googleapis.com/unknown'
    setup_run_circuit_with_result_(client, result)
    engine = cg.Engine(project_id='project-id', proto_version=cg.engine.engine.ProtoVersion.V2)
    job = engine.run_sweep(program=_CIRCUIT, job_id='job-id', params=cirq.Points('a', [1, 2]))
    with pytest.raises(ValueError, match='invalid result proto version'):
        job.results()