from typing import Iterable, List
from unittest import mock
import numpy as np
import pytest
import cirq
from cirq_google.line.placement.anneal import (
from cirq_google.line.placement.chip import chip_as_adjacency_list
def test_normalize_edge_normalizes():
    q00, q01 = (cirq.GridQubit(0, 0), cirq.GridQubit(0, 1))
    q10, q11 = (cirq.GridQubit(1, 0), cirq.GridQubit(1, 1))
    search = AnnealSequenceSearch(_create_device([]), seed=4027383826)
    assert search._normalize_edge((q00, q01)) == (q00, q01)
    assert search._normalize_edge((q01, q00)) == (q00, q01)
    assert search._normalize_edge((q01, q10)) == (q01, q10)
    assert search._normalize_edge((q10, q01)) == (q01, q10)
    assert search._normalize_edge((q00, q11)) == (q00, q11)
    assert search._normalize_edge((q11, q00)) == (q00, q11)