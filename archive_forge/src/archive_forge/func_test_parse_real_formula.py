import numpy as np
import pytest
import sympy
from cirq.interop.quirk.cells.parse import parse_matrix, parse_formula, parse_complex
def test_parse_real_formula():
    t = sympy.Symbol('t')
    assert parse_formula('1/2') == 0.5
    assert parse_formula('t*t + ln(t)') == t * t + sympy.ln(t)
    assert parse_formula('cos(pi*t)') == sympy.cos(sympy.pi * t)
    assert parse_formula('5t') == 5.0 * t
    np.testing.assert_allclose(parse_formula('cos(pi)'), -1, atol=1e-08)
    assert type(parse_formula('cos(pi)')) is float
    with pytest.raises(ValueError, match='real result'):
        _ = parse_formula('i')