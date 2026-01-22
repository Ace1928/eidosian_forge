import pennylane as qml
from pennylane.wires import Wires
def wires_ring(wires):
    """Wire sequence for the ring pattern"""
    if len(wires) in [0, 1]:
        return []
    if len(wires) == 2:
        return [wires.subset([0, 1])]
    sequence = [wires.subset([i, i + 1], periodic_boundary=True) for i in range(len(wires))]
    return sequence