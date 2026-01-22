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
def test_get_circuit_unsupported_program_type(get_program_async):
    program = cg.EngineProgram('a', 'b', EngineContext())
    get_program_async.return_value = quantum.QuantumProgram(code=any_pb2.Any(type_url='type.googleapis.com/unknown.proto'))
    with pytest.raises(ValueError, match='unknown.proto'):
        program.get_circuit()