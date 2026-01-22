import duet
import pytest
import cirq
def test_collect_with_reaction():
    events = [0]
    sent = 0
    received = 0

    class TestCollector(cirq.Collector):

        def next_job(self):
            nonlocal sent
            if sent >= received + 3:
                return None
            sent += 1
            events.append(sent)
            q = cirq.LineQubit(0)
            circuit = cirq.Circuit(cirq.H(q), cirq.measure(q))
            return cirq.CircuitSampleJob(circuit=circuit, repetitions=10, tag=sent)

        def on_job_result(self, job, result):
            nonlocal received
            received += 1
            events.append(-job.tag)
    TestCollector().collect(sampler=cirq.Simulator(), max_total_samples=100, concurrency=5)
    assert sorted(events) == list(range(-10, 1 + 10))
    assert [e for e in events if e > 0] == list(range(1, 11))
    assert all((events.index(-k) > events.index(k) for k in range(1, 11)))