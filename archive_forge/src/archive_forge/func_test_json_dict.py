from datetime import timedelta
import pytest
import sympy
import numpy as np
import cirq
from cirq.value import Duration
def test_json_dict():
    d = Duration(picos=6)
    assert d._json_dict_() == {'picos': 6}