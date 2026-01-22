from typing import Iterable, List
from unittest import mock
import numpy as np
import pytest
import cirq
from cirq_google.line.placement.anneal import (
from cirq_google.line.placement.chip import chip_as_adjacency_list
def test_force_edges_active_move_calls_force_edge_active_move():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q10 = cirq.GridQubit(1, 0)
    q11 = cirq.GridQubit(1, 1)
    search = AnnealSequenceSearch(_create_device([q00, q01, q10, q11]), seed=4027383813)
    with mock.patch.object(search, '_force_edge_active_move') as force_edge_active_move:
        search._force_edges_active_move(search._create_initial_solution())
        force_edge_active_move.assert_called_with(mock.ANY)