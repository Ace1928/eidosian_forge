import numpy as np
import pytest
import sympy
from cirq.interop.quirk.cells.parse import parse_matrix, parse_formula, parse_complex
def test_parse_complex_expression_failures():
    with pytest.raises(ValueError, match='Incomplete expression'):
        _ = parse_formula('(')
    with pytest.raises(ValueError, match="unmatched '\\)'"):
        _ = parse_formula(')')
    with pytest.raises(ValueError, match='binary op in bad spot'):
        _ = parse_formula('5+(/)')
    with pytest.raises(ValueError, match='operated on nothing'):
        _ = parse_formula('(5+)')
    with pytest.raises(ValueError, match='operated on nothing'):
        _ = parse_formula('(5/)')
    with pytest.raises(ValueError, match='binary op in bad spot'):
        _ = parse_formula('5-/2')
    with pytest.raises(ValueError, match='binary op in bad spot'):
        _ = parse_formula('/2')
    assert parse_formula('2/ ') == 2
    assert parse_formula('2* ') == 2
    assert parse_formula('2+ ') == 2
    assert parse_formula('2- ') == 2
    assert parse_formula('2^ ') == 2