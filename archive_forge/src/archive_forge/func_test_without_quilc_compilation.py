from typing import Tuple, Any, List, Union, Dict
import pytest
import cirq
from pyquil import Program
import numpy as np
import sympy
from cirq_rigetti import circuit_sweep_executors as executors, circuit_transformers
def test_without_quilc_compilation(mock_qpu_implementer: Any, parametric_circuit_with_params: Tuple[cirq.Circuit, cirq.Sweepable]) -> None:
    """test execution without quilc compilation treats the transformed cirq
    Circuit as native quil and does not pass it through quilc.
    """
    parametric_circuit, sweepable = parametric_circuit_with_params
    repetitions = 2
    param_resolvers = [r for r in cirq.to_resolvers(sweepable)]
    expected_results = [np.ones((repetitions,)) * (params['t'] if 't' in params else i) for i, params in enumerate(param_resolvers)]
    quantum_computer = mock_qpu_implementer.implement_passive_quantum_computer_with_results(expected_results)
    results = executors.without_quilc_compilation(quantum_computer=quantum_computer, circuit=parametric_circuit, resolvers=param_resolvers, repetitions=repetitions)
    assert len(param_resolvers) == len(results)
    assert 0 == quantum_computer.compiler.quil_to_native_quil.call_count
    assert len(param_resolvers) == quantum_computer.compiler.native_quil_to_executable.call_count
    for i, result in enumerate(results):
        result = results[i]
        assert param_resolvers[i] == result.params
        assert np.allclose(result.measurements['m'], expected_results[i]), 'should return an ordered list of results with correct set of measurements'