from typing import (
import cmath
import re
import numpy as np
import sympy
def parse_formula(formula: str) -> Union[float, sympy.Expr]:
    """Attempts to parse formula text in exactly the same way as Quirk."""
    if not isinstance(formula, str):
        raise TypeError('formula must be a string')
    token_map: Dict[str, _HangingToken] = {**PARSE_COMPLEX_TOKEN_MAP_RAD, 't': sympy.Symbol('t')}
    result = _parse_formula_using_token_map(formula, token_map)
    if isinstance(result, sympy.Basic):
        if result.free_symbols:
            return result
        result = complex(result)
    if isinstance(result, complex):
        if abs(np.imag(result)) > 1e-08:
            raise ValueError('Not a real result.')
        result = np.real(result)
    return float(cast(SupportsFloat, result))