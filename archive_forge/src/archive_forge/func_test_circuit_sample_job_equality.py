import duet
import pytest
import cirq
def test_circuit_sample_job_equality():
    eq = cirq.testing.EqualsTester()
    c1 = cirq.Circuit()
    c2 = cirq.Circuit(cirq.measure(cirq.LineQubit(0)))
    eq.add_equality_group(cirq.CircuitSampleJob(c1, repetitions=10), cirq.CircuitSampleJob(c1, repetitions=10, tag=None))
    eq.add_equality_group(cirq.CircuitSampleJob(c2, repetitions=10))
    eq.add_equality_group(cirq.CircuitSampleJob(c1, repetitions=100))
    eq.add_equality_group(cirq.CircuitSampleJob(c1, repetitions=10, tag='test'))