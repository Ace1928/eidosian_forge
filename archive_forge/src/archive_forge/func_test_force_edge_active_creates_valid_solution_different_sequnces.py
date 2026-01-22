from typing import Iterable, List
from unittest import mock
import numpy as np
import pytest
import cirq
from cirq_google.line.placement.anneal import (
from cirq_google.line.placement.chip import chip_as_adjacency_list
def test_force_edge_active_creates_valid_solution_different_sequnces():
    q00, q10, q20, q30 = [cirq.GridQubit(x, 0) for x in range(4)]
    q01, q11, q21, q31 = [cirq.GridQubit(x, 1) for x in range(4)]
    qubits = [q00, q10, q20, q30, q01, q11, q21, q31]
    search = AnnealSequenceSearch(_create_device(qubits), seed=4027383817)
    assert search._force_edge_active([[q00, q10, q20, q30], [q01, q11, q21, q31]], (q00, q01), lambda: True) == [[q30, q20, q10, q00, q01, q11, q21, q31]]
    assert search._force_edge_active([[q00, q10, q20, q30], [q01, q11, q21, q31]], (q30, q31), lambda: True) == [[q00, q10, q20, q30, q31, q21, q11, q01]]
    assert search._force_edge_active([[q00, q10, q20, q30], [q01, q11, q21, q31]], (q10, q11), lambda: True) == [[q30, q20, q10, q11, q21, q31], [q00], [q01]]
    assert search._force_edge_active([[q00, q10, q20, q30], [q01, q11, q21, q31]], (q10, q11), lambda: False) == [[q00, q10, q11, q01], [q20, q30], [q21, q31]]