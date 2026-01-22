import pytest
import cirq
def test_measurement_key_mapping():

    class MultiKeyGate:

        def __init__(self, keys):
            self._keys = frozenset(keys)

        def _measurement_key_names_(self):
            return self._keys

        def _with_measurement_key_mapping_(self, key_map):
            if not all((key in key_map for key in self._keys)):
                raise ValueError('missing keys')
            return MultiKeyGate([key_map[key] for key in self._keys])
    assert cirq.measurement_key_names(MultiKeyGate([])) == set()
    assert cirq.measurement_key_names(MultiKeyGate(['a'])) == {'a'}
    mkg_ab = MultiKeyGate(['a', 'b'])
    assert cirq.measurement_key_names(mkg_ab) == {'a', 'b'}
    mkg_cd = cirq.with_measurement_key_mapping(mkg_ab, {'a': 'c', 'b': 'd'})
    assert cirq.measurement_key_names(mkg_cd) == {'c', 'd'}
    mkg_ac = cirq.with_measurement_key_mapping(mkg_ab, {'a': 'a', 'b': 'c'})
    assert cirq.measurement_key_names(mkg_ac) == {'a', 'c'}
    mkg_ba = cirq.with_measurement_key_mapping(mkg_ab, {'a': 'b', 'b': 'a'})
    assert cirq.measurement_key_names(mkg_ba) == {'a', 'b'}
    with pytest.raises(ValueError):
        cirq.with_measurement_key_mapping(mkg_ab, {'a': 'c'})
    assert cirq.with_measurement_key_mapping(cirq.X, {'a': 'c'}) is NotImplemented
    mkg_cdx = cirq.with_measurement_key_mapping(mkg_ab, {'a': 'c', 'b': 'd', 'x': 'y'})
    assert cirq.measurement_key_names(mkg_cdx) == {'c', 'd'}