import dataclasses
import cirq
import cirq_google
import pytest
from cirq_google import (
def test_kv_executable_spec():
    kv1 = KeyValueExecutableSpec.from_dict(dict(name='test', idx=5), executable_family='cirq_google.algo_benchmarks.example')
    kv2 = KeyValueExecutableSpec(executable_family='cirq_google.algo_benchmarks.example', key_value_pairs=(('name', 'test'), ('idx', 5)))
    assert kv1 == kv2
    assert hash(kv1) == hash(kv2)
    with pytest.raises(TypeError, match='unhashable.*'):
        hash(KeyValueExecutableSpec(executable_family='', key_value_pairs=[('name', 'test')]))