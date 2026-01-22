from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
def test_gate_json_dict():
    g = cirq.CSWAP
    assert g._json_dict_() == {}