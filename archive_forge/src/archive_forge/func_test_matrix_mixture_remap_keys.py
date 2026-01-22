import cirq
import numpy as np
import pytest
def test_matrix_mixture_remap_keys():
    dp = cirq.depolarize(0.1)
    mm = cirq.MixedUnitaryChannel.from_mixture(dp)
    with pytest.raises(TypeError):
        _ = cirq.measurement_key_name(mm)
    assert cirq.with_measurement_key_mapping(mm, {'a': 'b'}) is NotImplemented
    mm_x = cirq.MixedUnitaryChannel.from_mixture(dp, key='x')
    assert cirq.with_measurement_key_mapping(mm_x, {'a': 'b'}) is mm_x
    assert cirq.measurement_key_name(cirq.with_key_path(mm_x, ('path',))) == 'path:x'
    mm_a = cirq.MixedUnitaryChannel.from_mixture(dp, key='a')
    mm_b = cirq.MixedUnitaryChannel.from_mixture(dp, key='b')
    assert mm_a != mm_b
    assert cirq.with_measurement_key_mapping(mm_a, {'a': 'b'}) == mm_b