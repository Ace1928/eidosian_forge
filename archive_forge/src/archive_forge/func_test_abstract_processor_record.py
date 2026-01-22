from unittest import mock
import pytest
import cirq
import cirq_google as cg
def test_abstract_processor_record():
    proc_rec = _ExampleProcessorRecord()
    assert isinstance(proc_rec.get_processor(), cg.engine.AbstractProcessor)
    assert isinstance(proc_rec.get_sampler(), cirq.Sampler)
    assert isinstance(proc_rec.get_device(), cirq.Device)