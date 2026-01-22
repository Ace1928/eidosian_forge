from typing import Any, List, Sequence, Tuple
import cirq
import pytest
from pyquil import Program
from pyquil.api import QuantumComputer
import numpy as np
from pyquil.gates import MEASURE, RX, X, DECLARE, H, CNOT
from cirq_rigetti import RigettiQCSService
from typing_extensions import Protocol
from cirq_rigetti import circuit_transformers as transformers
from cirq_rigetti import circuit_sweep_executors as executors
@pytest.mark.parametrize('result_builder', [_build_service_results, _build_sampler_results])
def test_bell_circuit(mock_qpu_implementer: Any, bell_circuit: cirq.Circuit, result_builder: _ResultBuilder) -> None:
    """test that RigettiQCSService and RigettiQCSSampler can run a basic Bell circuit
    with two read out bits and return expected cirq.Results.
    """
    param_resolvers = [cirq.ParamResolver({})]
    results, quantum_computer, expected_results, param_resolvers = result_builder(mock_qpu_implementer, bell_circuit, param_resolvers)
    assert len(param_resolvers) == len(results), 'should return a result for every element in sweepable'
    for i, param_resolver in enumerate(param_resolvers):
        result = results[i]
        assert param_resolver == result.params
        assert np.allclose(result.measurements['m'], expected_results[i]), 'should return an ordered list of results with correct set of measurements'

    def test_executable(program: Program) -> None:
        assert H(0) in program.instructions, 'bell circuit should include Hadamard'
        assert CNOT(0, 1) in program.instructions, 'bell circuit should include CNOT'
        assert DECLARE('m0', memory_size=2) in program.instructions, 'executable should declare a read out bit'
        assert MEASURE(0, ('m0', 0)) in program.instructions, 'executable should measure the first qubit to the first read out bit'
        assert MEASURE(1, ('m0', 1)) in program.instructions, 'executable should measure the second qubit to the second read out bit'
    param_sweeps = len(param_resolvers)
    assert param_sweeps == quantum_computer.compiler.quil_to_native_quil.call_count
    for i, call_args in enumerate(quantum_computer.compiler.quil_to_native_quil.call_args_list):
        test_executable(call_args[0][0])
    assert param_sweeps == quantum_computer.compiler.native_quil_to_executable.call_count
    for i, call_args in enumerate(quantum_computer.compiler.native_quil_to_executable.call_args_list):
        test_executable(call_args[0][0])
    assert param_sweeps == quantum_computer.qam.run.call_count
    for i, call_args in enumerate(quantum_computer.qam.run.call_args_list):
        test_executable(call_args[0][0])