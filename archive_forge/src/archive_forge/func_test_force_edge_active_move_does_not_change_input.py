from typing import Iterable, List
from unittest import mock
import numpy as np
import pytest
import cirq
from cirq_google.line.placement.anneal import (
from cirq_google.line.placement.chip import chip_as_adjacency_list
def test_force_edge_active_move_does_not_change_input():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q10 = cirq.GridQubit(1, 0)
    q11 = cirq.GridQubit(1, 1)
    search = AnnealSequenceSearch(_create_device([q00, q01, q10, q11]), seed=4027383814)
    seqs, edges = search._create_initial_solution()
    seqs_copy, edges_copy = (list(seqs), edges.copy())
    search._force_edge_active_move((seqs, edges))
    assert seqs_copy == seqs
    assert edges_copy == edges