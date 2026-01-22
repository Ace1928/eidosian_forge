import pytest
import cirq
import networkx as nx
def test_griddevice_self_loop():
    bad_pairs = [(cirq.GridQubit(0, 0), cirq.GridQubit(0, 0)), (cirq.GridQubit(1, 0), cirq.GridQubit(1, 1))]
    with pytest.raises(ValueError, match='Self loop'):
        _ = cirq.GridDeviceMetadata(bad_pairs, cirq.Gateset(cirq.XPowGate))