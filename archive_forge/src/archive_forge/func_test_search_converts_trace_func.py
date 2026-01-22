from typing import Iterable, List
from unittest import mock
import numpy as np
import pytest
import cirq
from cirq_google.line.placement.anneal import (
from cirq_google.line.placement.chip import chip_as_adjacency_list
@mock.patch('cirq_google.line.placement.optimization.anneal_minimize')
def test_search_converts_trace_func(anneal_minimize):
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    seqs = [[q00, q01]]
    edges = {(q00, q01)}
    anneal_minimize.return_value = (seqs, edges)
    trace_func = mock.Mock()
    assert AnnealSequenceSearch(_create_device([]), seed=4027383810).search(trace_func=trace_func) == seqs
    wrapper_func = anneal_minimize.call_args[1]['trace_func']
    wrapper_func((seqs, edges), 1.0, 2.0, 3.0, True)
    trace_func.assert_called_once_with(seqs, 1.0, 2.0, 3.0, True)