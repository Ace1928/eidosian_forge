import numpy as np
import pytest
import sympy
from cirq.interop.quirk.cells.parse import parse_matrix, parse_formula, parse_complex
def test_parse_formula_failures():
    with pytest.raises(TypeError, match='formula must be a string'):
        _ = parse_formula(2)
    with pytest.raises(TypeError, match='formula must be a string'):
        _ = parse_formula([])
    with pytest.raises(ValueError, match='Unrecognized token'):
        _ = parse_formula('5*__**DSA **)SADD')
    with pytest.raises(ValueError, match='Unrecognized token'):
        _ = parse_formula('5*x')