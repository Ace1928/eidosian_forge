from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
from cirq_aqt import AQTSampler, AQTSamplerLocalSimulator
from cirq_aqt.aqt_device import get_aqt_device, get_op_string
def test_aqt_sampler_error_handling():
    for e_return in [EngineError(), EngineErrorSecond(), EngineNoStatus(), EngineNoStatus2(), EngineNoid()]:
        with mock.patch('cirq_aqt.aqt_sampler.put', return_value=e_return, side_effect=e_return.update) as _mock_method:
            theta = sympy.Symbol('theta')
            num_points = 1
            max_angle = np.pi
            repetitions = 10
            sampler = AQTSampler(remote_host='http://localhost:5000', access_token='testkey')
            _, qubits = get_aqt_device(1)
            circuit = cirq.Circuit(cirq.X(qubits[0]) ** theta)
            sweep = cirq.Linspace(key='theta', start=0.1, stop=max_angle / np.pi, length=num_points)
            with pytest.raises(RuntimeError):
                _results = sampler.run_sweep(circuit, params=sweep, repetitions=repetitions)