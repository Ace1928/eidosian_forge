import numpy as np
import pytest
import sympy
from cirq.interop.quirk.cells.parse import parse_matrix, parse_formula, parse_complex
def test_parse_matrix_failures():
    with pytest.raises(ValueError, match='Not surrounded by {{}}'):
        _ = parse_matrix('1')
    with pytest.raises(ValueError, match='Not surrounded by {{}}'):
        _ = parse_matrix('{{1}')
    with pytest.raises(ValueError, match='Not surrounded by {{}}'):
        _ = parse_matrix('{1}}')
    with pytest.raises(ValueError, match='Not surrounded by {{}}'):
        _ = parse_matrix('1}}')
    with pytest.raises(ValueError, match='Failed to parse complex'):
        _ = parse_matrix('{{x}}')