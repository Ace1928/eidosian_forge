from typing import cast
import numpy as np
import pytest
import cirq
from cirq.interop.quirk.cells import arithmetic_cells
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
from cirq import quirk_url_to_circuit
def test_arithmetic_modular_multiply_gates():
    assert_url_to_circuit_returns('{"cols":[[{"id":"setA","arg":3},{"id":"setR","arg":7}],["*AmodR4"]]}', maps={0: 0, 1: 3, 3: 2, 2: 6, 6: 4, 4: 5, 5: 1, 7: 7, 15: 15})
    assert_url_to_circuit_returns('{"cols":[[{"id":"setA","arg":3},{"id":"setR","arg":7}],["/AmodR4"]]}', maps={0: 0, 1: 5, 2: 3, 3: 1, 4: 6, 5: 4, 6: 2, 7: 7, 15: 15})
    assert_url_to_circuit_returns('{"cols":[[{"id":"setA","arg":5},{"id":"setR","arg":15}],["*AmodR4"]]}', maps={0: 0, 1: 1, 3: 3, 15: 15})
    assert_url_to_circuit_returns('{"cols":[[{"id":"setA","arg":5},{"id":"setR","arg":15}],["/AmodR4"]]}', maps={0: 0, 1: 1, 3: 3, 15: 15})