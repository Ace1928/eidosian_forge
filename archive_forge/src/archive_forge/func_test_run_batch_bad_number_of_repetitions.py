from unittest import mock
import pytest
import cirq
import cirq_google as cg
from cirq_google.engine.abstract_processor import AbstractProcessor
def test_run_batch_bad_number_of_repetitions():
    processor = mock.create_autospec(AbstractProcessor)
    sampler = cg.ProcessorSampler(processor=processor)
    a = cirq.LineQubit(0)
    circuit1 = cirq.Circuit(cirq.X(a))
    circuit2 = cirq.Circuit(cirq.Y(a))
    params1 = [cirq.ParamResolver({'t': 1})]
    params2 = [cirq.ParamResolver({'t': 2})]
    circuits = [circuit1, circuit2]
    params_list = [params1, params2]
    with pytest.raises(ValueError, match='2 and 3'):
        sampler.run_batch(circuits, params_list, [5, 5, 5])