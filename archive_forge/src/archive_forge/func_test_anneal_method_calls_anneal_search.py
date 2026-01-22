from unittest import mock
import cirq
import cirq_google as cg
from cirq_google import line_on_device
from cirq_google.line.placement import GridQubitLineTuple
def test_anneal_method_calls_anneal_search():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q03 = cirq.GridQubit(0, 3)
    device = FakeDevice(qubits=[q00, q01, q03])
    length = 2
    method = cg.AnnealSequenceSearchStrategy()
    with mock.patch.object(method, 'place_line') as place_line:
        sequences = GridQubitLineTuple((q00, q01))
        place_line.return_value = sequences
        assert line_on_device(device, length, method) == sequences
        place_line.assert_called_once_with(device, length)