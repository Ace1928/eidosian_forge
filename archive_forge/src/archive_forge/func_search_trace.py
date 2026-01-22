from typing import Callable, List, Optional, Tuple, Set, Any, TYPE_CHECKING
import numpy as np
import cirq
from cirq_google.line.placement import place_strategy, optimization
from cirq_google.line.placement.chip import above, right_of, chip_as_adjacency_list, EDGE
from cirq_google.line.placement.sequence import GridQubitLineTuple, LineSequence
def search_trace(state: _STATE, temp: float, cost: float, probability: float, accepted: bool):
    if trace_func:
        trace_seqs, _ = state
        trace_func(trace_seqs, temp, cost, probability, accepted)