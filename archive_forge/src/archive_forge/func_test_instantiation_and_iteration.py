from typing import Dict, List, TYPE_CHECKING
import pytest
import numpy as np
import cirq
from cirq_google.engine.abstract_job import AbstractJob
def test_instantiation_and_iteration():
    job = MockJob()
    assert len(job) == 5
    assert job[3].measurements['a'][0] == 3
    count = 0
    for result in job:
        assert result.measurements['a'][0] == count
        count += 1
    iterator = iter(job)
    result = next(iterator)
    assert result.measurements['a'][0] == 0
    result = next(iterator)
    assert result.measurements['a'][0] == 1
    result = next(iterator)
    assert result.measurements['a'][0] == 2
    result = next(iterator)
    assert result.measurements['a'][0] == 3
    result = next(iterator)
    assert result.measurements['a'][0] == 4
    with pytest.raises(StopIteration):
        next(iterator)