import re
import pytest
import cirq
def test_record_measurement():
    cd = cirq.ClassicalDataDictionaryStore()
    cd.record_measurement(mkey_m, (0, 1), two_qubits)
    assert cd.records == {mkey_m: [(0, 1)]}
    assert cd.keys() == (mkey_m,)
    assert cd.measured_qubits == {mkey_m: [two_qubits]}