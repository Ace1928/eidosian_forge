import duet
import pytest
import cirq
def test_circuit_sample_job_repr():
    cirq.testing.assert_equivalent_repr(cirq.CircuitSampleJob(cirq.Circuit(cirq.H(cirq.LineQubit(0))), repetitions=10, tag='guess'))