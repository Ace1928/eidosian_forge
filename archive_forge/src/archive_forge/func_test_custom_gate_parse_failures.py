import json
import urllib
import numpy as np
import pytest
import cirq
from cirq import quirk_url_to_circuit, quirk_json_to_circuit
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
def test_custom_gate_parse_failures():
    with pytest.raises(ValueError, match='must be a list'):
        _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":[],"gates":5}')
    with pytest.raises(ValueError, match='gate json must be a dict'):
        _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":[],"gates":[5]}')
    with pytest.raises(ValueError, match='Circuit JSON must be a dict'):
        _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":[],"gates":[{"id":"~a","circuit":5}]}')
    with pytest.raises(ValueError, match='matrix json must be a string'):
        _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":[],"gates":[{"id":"~a","matrix":5}]}')
    with pytest.raises(ValueError, match='Not surrounded by {{}}'):
        _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":[],"gates":[{"id":"~a","matrix":"abc"}]}')
    with pytest.raises(ValueError, match='must have an id'):
        _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":[],"gates":[{"matrix":"{{1,0},{0,1}}"}]}')
    with pytest.raises(ValueError, match='both a matrix and a circuit'):
        _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":[],"gates":[{"id":"~a","circuit":{"cols":[]},"matrix":"{{1,0},{0,1}}"}]}')
    with pytest.raises(ValueError, match='matrix or a circuit'):
        _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":[],"gates":[{"id":"~a"}]}')
    with pytest.raises(ValueError, match='duplicate identifier'):
        _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":[],"gates":[{"id":"~a","matrix":"{{1,0},{0,1}}"},{"id":"~a","matrix":"{{1,0},{0,1}}"}]}')