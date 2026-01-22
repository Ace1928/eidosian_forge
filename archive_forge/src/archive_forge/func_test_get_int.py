import re
import pytest
import cirq
def test_get_int():
    cd = cirq.ClassicalDataDictionaryStore()
    cd.record_measurement(mkey_m, (0, 1), two_qubits)
    assert cd.get_int(mkey_m) == 1
    cd = cirq.ClassicalDataDictionaryStore()
    cd.record_measurement(mkey_m, (1, 1), two_qubits)
    assert cd.get_int(mkey_m) == 3
    cd = cirq.ClassicalDataDictionaryStore()
    cd.record_channel_measurement(mkey_m, 1)
    assert cd.get_int(mkey_m) == 1
    cd = cirq.ClassicalDataDictionaryStore()
    cd.record_measurement(mkey_m, (1, 1), cirq.LineQid.range(2, dimension=3))
    assert cd.get_int(mkey_m) == 4
    cd = cirq.ClassicalDataDictionaryStore()
    with pytest.raises(KeyError, match='The measurement key m is not in {}'):
        cd.get_int(mkey_m)