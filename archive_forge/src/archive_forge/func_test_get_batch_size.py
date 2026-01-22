import datetime
from unittest import mock
import pytest
import numpy as np
from google.protobuf import any_pb2, timestamp_pb2
from google.protobuf.text_format import Merge
import cirq
import cirq_google as cg
from cirq_google.api import v1, v2
from cirq_google.engine import util
from cirq_google.cloud import quantum
from cirq_google.engine.engine import EngineContext
from cirq_google.engine.result_type import ResultType
@mock.patch('cirq_google.engine.engine_client.EngineClient.get_program_async')
def test_get_batch_size(get_program_async):
    program = cg.EngineProgram('a', 'b', EngineContext(), result_type=ResultType.Batch)
    get_program_async.return_value = quantum.QuantumProgram(code=_BATCH_PROGRAM_V2)
    assert program.batch_size() == 1
    program = cg.EngineProgram('a', 'b', EngineContext(), _program=quantum.QuantumProgram(code=_BATCH_PROGRAM_V2), result_type=ResultType.Batch)
    assert program.batch_size() == 1
    with pytest.raises(ValueError, match='ResultType.Program'):
        program = cg.EngineProgram('a', 'b', EngineContext(), result_type=ResultType.Program)
        _ = program.batch_size()
    with pytest.raises(ValueError, match='cirq.google.api.v2.Program'):
        get_program_async.return_value = quantum.QuantumProgram(code=_PROGRAM_V2)
        program = cg.EngineProgram('a', 'b', EngineContext(), result_type=ResultType.Batch)
        _ = program.batch_size()