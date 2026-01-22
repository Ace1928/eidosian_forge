import re
import pytest
import cirq
def test_record_measurement_errors():
    cd = cirq.ClassicalDataDictionaryStore()
    with pytest.raises(ValueError, match='3 measurements but 2 qubits'):
        cd.record_measurement(mkey_m, (0, 1, 2), two_qubits)
    cd.record_measurement(mkey_m, (0, 1), two_qubits)
    cd.record_measurement(mkey_m, (1, 0), two_qubits)
    with pytest.raises(ValueError, match=re.escape('Measurement shape (2, 2, 2) does not match (2, 2) in m')):
        cd.record_measurement(mkey_m, (1, 0, 4), tuple(cirq.LineQubit.range(3)))
    with pytest.raises(ValueError, match=re.escape('Measurement shape (3, 3) does not match (2, 2) in m')):
        cd.record_measurement(mkey_m, (1, 0), tuple(cirq.LineQid.range(2, dimension=3)))