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
def test_sampler_with_unary_rpcs(client):
    setup_run_circuit_with_result_(client, _RESULTS)
    engine = cg.Engine(project_id='proj', context=EngineContext(enable_streaming=False))
    sampler = engine.get_sampler(processor_id='tmp')
    results = sampler.run_sweep(program=_CIRCUIT, params=[cirq.ParamResolver({'a': 1}), cirq.ParamResolver({'a': 2})])
    assert len(results) == 2
    for i, v in enumerate([1, 2]):
        assert results[i].repetitions == 1
        assert results[i].params.param_dict == {'a': v}
        assert results[i].measurements == {'q': np.array([[0]], dtype='uint8')}
    assert client().create_program_async.call_args[0][0] == 'proj'
    with cirq.testing.assert_deprecated('sampler', deadline='1.0'):
        _ = engine.sampler(processor_id='tmp')
    with pytest.raises(ValueError, match='list of processors'):
        _ = engine.get_sampler(['test1', 'test2'])