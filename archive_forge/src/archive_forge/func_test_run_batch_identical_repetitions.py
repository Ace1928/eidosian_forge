from unittest import mock
import pytest
import cirq
import cirq_google as cg
from cirq_google.engine.abstract_processor import AbstractProcessor
@pytest.mark.parametrize('run_name, device_config_name', [('run_name', 'device_config_alias'), ('', '')])
def test_run_batch_identical_repetitions(run_name, device_config_name):
    processor = mock.create_autospec(AbstractProcessor)
    sampler = cg.ProcessorSampler(processor=processor, run_name=run_name, device_config_name=device_config_name)
    a = cirq.LineQubit(0)
    circuit1 = cirq.Circuit(cirq.X(a))
    circuit2 = cirq.Circuit(cirq.Y(a))
    params1 = [cirq.ParamResolver({'t': 1})]
    params2 = [cirq.ParamResolver({'t': 2})]
    sampler.run_batch([circuit1, circuit2], [params1, params2], [5, 5])
    expected_calls = [mock.call(program=circuit1, params=params1, repetitions=5, run_name=run_name, device_config_name=device_config_name), mock.call().results_async(), mock.call(program=circuit2, params=params2, repetitions=5, run_name=run_name, device_config_name=device_config_name), mock.call().results_async()]
    processor.run_sweep_async.assert_has_calls(expected_calls)