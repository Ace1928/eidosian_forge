import networkx as nx
import cirq
def test_device_metadata():

    class RawDevice(cirq.Device):
        pass
    assert RawDevice().metadata is None