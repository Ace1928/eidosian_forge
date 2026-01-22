from typing import Dict, List, Optional, Set, TYPE_CHECKING
import abc
import collections
from cirq.devices import GridQubit
from cirq_google.line.placement import place_strategy
from cirq_google.line.placement.chip import chip_as_adjacency_list
from cirq_google.line.placement.sequence import GridQubitLineTuple
def place_line(self, device: 'cirq_google.GridDevice', length: int) -> GridQubitLineTuple:
    """Runs line sequence search.

        Args:
            device: Chip description.
            length: Required line length.

        Returns:
            Linear sequences found on the chip.

        Raises:
            ValueError: If search algorithm passed on initialization is not
                        recognized.
        """
    if not device.metadata.qubit_set:
        return GridQubitLineTuple()
    start: GridQubit = min(device.metadata.qubit_set)
    sequences: List[LineSequence] = []
    greedy_search: Dict[str, List[GreedySequenceSearch]] = {'minimal_connectivity': [_PickFewestNeighbors(device, start)], 'largest_area': [_PickLargestArea(device, start)], 'best': [_PickFewestNeighbors(device, start), _PickLargestArea(device, start)]}
    algos = greedy_search.get(self.algorithm)
    if algos is None:
        raise ValueError(f'Unknown greedy search algorithm {self.algorithm}')
    for algorithm in algos:
        sequences.append(algorithm.get_or_search())
    return GridQubitLineTuple.best_of(sequences, length)