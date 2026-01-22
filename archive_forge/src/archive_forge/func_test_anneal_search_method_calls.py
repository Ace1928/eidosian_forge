from typing import Iterable, List
from unittest import mock
import numpy as np
import pytest
import cirq
from cirq_google.line.placement.anneal import (
from cirq_google.line.placement.chip import chip_as_adjacency_list
def test_anneal_search_method_calls():
    q00, q01 = (cirq.GridQubit(0, 0), cirq.GridQubit(0, 1))
    device = _create_device([q00, q01])
    length = 1
    seed = 1
    method = AnnealSequenceSearchStrategy(None, seed)
    assert len(method.place_line(device, length)) == length