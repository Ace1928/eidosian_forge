from typing import Iterable, List
from unittest import mock
import numpy as np
import pytest
import cirq
from cirq_google.line.placement.anneal import (
from cirq_google.line.placement.chip import chip_as_adjacency_list
def test_force_edge_active_creates_valid_solution_single_sequence():
    q00, q10, q20, q30 = [cirq.GridQubit(x, 0) for x in range(4)]
    q01, q11, q21, q31 = [cirq.GridQubit(x, 1) for x in range(4)]
    c = [q00, q10, q20, q30, q01, q11, q21, q31]
    search = AnnealSequenceSearch(_create_device(c), seed=4027383824)
    assert search._force_edge_active([[q30, q20, q10, q00, q01, q11, q21, q31]], (q30, q31), lambda: True) == [[q20, q10, q00, q01, q11, q21, q31, q30]]
    assert search._force_edge_active([[q31, q21, q11, q01, q00, q10, q20, q30]], (q30, q31), lambda: True) == [[q21, q11, q01, q00, q10, q20, q30, q31]]
    assert search._force_edge_active([[q30, q20, q10, q00, q01, q11, q21, q31]], (q30, q31), lambda: False) == [[q30, q31], [q20, q10, q00, q01, q11, q21]]
    assert search._force_edge_active([[q30, q20, q10, q00, q01, q11, q21, q31]], (q20, q21), lambda: True) == [[q10, q00, q01, q11, q21, q20, q30], [q31]]
    assert search._force_edge_active([[q30, q20, q10, q00, q01, q11, q21, q31]], (q20, q21), lambda: False) == [[q30, q20, q21, q31], [q10, q00, q01, q11]]
    assert search._force_edge_active([[q30, q20, q10, q00, q01, q11, q21], [q31]], (q20, q21), lambda: True) == [[q31], [q10, q00, q01, q11, q21, q20, q30]]
    assert search._force_edge_active([[q30, q20, q10, q00, q01, q11, q21], [q31]], (q20, q21), lambda: False) == [[q31], [q30, q20, q21], [q10, q00, q01, q11]]
    assert search._force_edge_active([[q20, q10, q00, q01, q11, q21, q31], [q30]], (q20, q21), lambda: True) == [[q30], [q10, q00, q01, q11, q21, q20], [q31]]
    samples = iter([True, False])
    assert search._force_edge_active([[q20, q10, q00, q01, q11, q21, q31], [q30]], (q20, q21), lambda: next(samples)) == [[q30], [q31, q21, q20, q10, q00, q01, q11]]
    assert search._force_edge_active([[q20, q10, q00, q01, q11, q21, q31], [q30]], (q20, q21), lambda: False) == [[q30], [q20, q21, q31], [q10, q00, q01, q11]]