from unittest import mock
import pytest
import cirq
import cirq_google as cg
def test_simulated_backend_descriptive_name():
    p = cg.SimulatedProcessorWithLocalDeviceRecord('rainbow')
    assert str(p) == 'rainbow-simulator'
    assert isinstance(p.get_sampler(), cg.engine.ProcessorSampler)
    assert isinstance(p.get_sampler().processor._sampler._sampler, cirq.Simulator)
    p = cg.SimulatedProcessorWithLocalDeviceRecord('rainbow', noise_strength=0.001)
    assert str(p) == 'rainbow-depol(1.000e-03)'
    assert isinstance(p.get_sampler().processor._sampler._sampler, cirq.DensityMatrixSimulator)
    p = cg.SimulatedProcessorWithLocalDeviceRecord('rainbow', noise_strength=float('inf'))
    assert str(p) == 'rainbow-zeros'
    assert isinstance(p.get_sampler().processor._sampler._sampler, cirq.ZerosSampler)