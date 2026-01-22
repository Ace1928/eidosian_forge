import pytest
import numpy as np
import sympy
import cirq
from cirq_google.cloud import quantum
from cirq_google.engine.abstract_local_program import AbstractLocalProgram
from cirq_google.engine.simulated_local_job import SimulatedLocalJob
from cirq_google.engine.local_simulation_type import LocalSimulationType
def test_run_async():
    qubits = cirq.LineQubit.range(20)
    c = cirq.testing.random_circuit(qubits, n_moments=20, op_density=1.0)
    c.append(cirq.measure(*qubits))
    program = ParentProgram([c], None)
    job = SimulatedLocalJob(job_id='test_job', processor_id='test1', parent_program=program, repetitions=100, sweeps=[{}], simulation_type=LocalSimulationType.ASYNCHRONOUS)
    assert job.execution_status() == quantum.ExecutionStatus.State.RUNNING
    _ = job.results()
    assert job.execution_status() == quantum.ExecutionStatus.State.SUCCESS