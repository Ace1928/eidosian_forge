from unittest import mock
import pytest
import cirq
import cirq_google as cg
from cirq_google.engine.abstract_processor import AbstractProcessor
@pytest.mark.parametrize('circuit', [cirq.Circuit(), cirq.FrozenCircuit()])
@pytest.mark.parametrize('run_name, device_config_name', [('run_name', 'device_config_alias'), ('', '')])
def test_run_circuit(circuit, run_name, device_config_name):
    processor = mock.create_autospec(AbstractProcessor)
    sampler = cg.ProcessorSampler(processor=processor, run_name=run_name, device_config_name=device_config_name)
    params = [cirq.ParamResolver({'a': 1})]
    sampler.run_sweep(circuit, params, 5)
    processor.run_sweep_async.assert_called_with(params=params, program=circuit, repetitions=5, run_name=run_name, device_config_name=device_config_name)