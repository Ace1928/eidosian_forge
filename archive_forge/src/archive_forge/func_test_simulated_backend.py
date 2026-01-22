from unittest import mock
import pytest
import cirq
import cirq_google as cg
@mock.patch('cirq_google.cloud.quantum.QuantumEngineServiceClient')
@mock.patch('cirq_google.engine.engine_client.EngineClient.get_processor')
def test_simulated_backend(get_processor, _):
    _set_get_processor_return(get_processor)
    with mock.patch('google.auth.default', lambda: (None, 'project!')):
        proc_rec = cg.SimulatedProcessorRecord('rainbow')
        assert isinstance(proc_rec.get_processor(), cg.engine.AbstractProcessor)
        assert isinstance(proc_rec.get_sampler(), cirq.Sampler)
        assert isinstance(proc_rec.get_device(), cirq.Device)
    assert proc_rec.processor_id == 'rainbow'
    assert str(proc_rec) == 'rainbow-simulator'
    cirq.testing.assert_equivalent_repr(proc_rec, global_vals={'cirq_google': cg})