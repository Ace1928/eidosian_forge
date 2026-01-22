from typing import cast
import numpy as np
import pytest
import cirq
from cirq.interop.quirk.cells import arithmetic_cells
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
from cirq import quirk_url_to_circuit
def test_arithmetic_modular_exponentiation_gates():
    assert_url_to_circuit_returns('{"cols":[[{"id":"setA","arg":5},{"id":"setB","arg":3},{"id":"setR","arg":7}],["*BToAmodR4"]]}', maps={0: 0, 1: 5, 2: 3, 15: 15})
    assert_url_to_circuit_returns('{"cols":[[{"id":"setA","arg":6},{"id":"setB","arg":3},{"id":"setR","arg":7}],["*BToAmodR4"]]}', maps={0: 0, 1: 1, 2: 2, 15: 15})
    assert_url_to_circuit_returns('{"cols":[[{"id":"setA","arg":5},{"id":"setB","arg":3},{"id":"setR","arg":7}],["/BToAmodR4"]]}', maps={0: 0, 1: 3, 2: 6, 15: 15})
    assert_url_to_circuit_returns('{"cols":[[{"id":"setA","arg":6},{"id":"setB","arg":3},{"id":"setR","arg":7}],["/BToAmodR4"]]}', maps={0: 0, 1: 1, 2: 2, 15: 15})