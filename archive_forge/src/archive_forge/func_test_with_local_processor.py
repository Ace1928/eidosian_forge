from unittest import mock
import pytest
import cirq
import cirq_google as cg
from cirq_google.engine.abstract_processor import AbstractProcessor
def test_with_local_processor():
    sampler = cg.ProcessorSampler(processor=cg.engine.SimulatedLocalProcessor(processor_id='my-fancy-processor'))
    r = sampler.run(cirq.Circuit(cirq.measure(cirq.LineQubit(0), key='z')))
    assert isinstance(r, cg.EngineResult)
    assert r.job_id == 'projects/fake_project/processors/my-fancy-processor/job/2'
    assert r.measurements['z'] == [[0]]