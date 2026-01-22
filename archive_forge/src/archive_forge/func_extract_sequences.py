from typing import Callable, List, Optional, Tuple, Set, Any, TYPE_CHECKING
import numpy as np
import cirq
from cirq_google.line.placement import place_strategy, optimization
from cirq_google.line.placement.chip import above, right_of, chip_as_adjacency_list, EDGE
from cirq_google.line.placement.sequence import GridQubitLineTuple, LineSequence
def extract_sequences() -> List[List[cirq.GridQubit]]:
    """Creates list of sequences for initial state.

            Returns:
              List of lists of sequences constructed on the chip.
            """
    seqs = []
    prev = None
    seq = None
    for node in self._c:
        if prev is None:
            seq = [node]
        elif node in self._c_adj[prev]:
            seq.append(node)
        else:
            seqs.append(seq)
            seq = [node]
        prev = node
    if seq:
        seqs.append(seq)
    return seqs