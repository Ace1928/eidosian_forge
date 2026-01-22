import pytest
from pyparsing import ParseException
from ..parsing import (
from ..testing import requires
@requires(parsing_library)
@pytest.mark.parametrize('species, composition', [('H2O*', {1: 2, 8: 1}), ("H2O'", {1: 2, 8: 1}), ("H2O''", {1: 2, 8: 1}), ("H2O'''", {1: 2, 8: 1}), ('Na', {11: 1}), ('Na*', {11: 1}), ("Na'", {11: 1}), ("Na''", {11: 1}), ("Na'''", {11: 1}), ('Na(g)', {11: 1}), ('Na*(g)', {11: 1}), ("Na'(g)", {11: 1}), ("Na''(g)", {11: 1}), ("Na'''(g)", {11: 1}), ('Na+', {0: 1, 11: 1}), ('Na*+', {0: 1, 11: 1}), ("Na'+", {0: 1, 11: 1}), ("Na''+", {0: 1, 11: 1}), ("Na'''+", {0: 1, 11: 1}), ('Na+(g)', {0: 1, 11: 1}), ('Na*+(g)', {0: 1, 11: 1}), ("Na'+(g)", {0: 1, 11: 1}), ("Na''+(g)", {0: 1, 11: 1}), ("Na'''+(g)", {0: 1, 11: 1}), ('Na2CO3*', {11: 2, 6: 1, 8: 3}), ('Na2CO3..7H2O*(s)', {11: 2, 6: 1, 8: 10, 1: 14})])
def test_formula_to_composition_primes(species, composition):
    """Should parse special species."""
    assert formula_to_composition(species) == composition