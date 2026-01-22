from typing import TYPE_CHECKING
from cirq_google.line.placement import greedy
from cirq_google.line.placement.place_strategy import LinePlacementStrategy
from cirq_google.line.placement.sequence import GridQubitLineTuple
def line_on_device(device: 'cirq_google.GridDevice', length: int, method: LinePlacementStrategy=greedy.GreedySequenceSearchStrategy()) -> GridQubitLineTuple:
    """Searches for linear sequence of qubits on device.

    Args:
        device: Google Xmon device instance.
        length: Desired number of qubits making up the line.
        method: Line placement method. Defaults to
                cirq.greedy.GreedySequenceSearchMethod.

    Returns:
        Line sequences search results.
    """
    return method.place_line(device, length)