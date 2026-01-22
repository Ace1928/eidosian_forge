import pytest
from pyparsing import ParseException
from ..parsing import (
from ..testing import requires
@pytest.mark.parametrize('species, composition', [('Li@C60', {3: 1, 6: 60}), ('Li@C60Cl', {3: 1, 6: 60, 17: 1}), ('(Li@C60)+', {0: 1, 3: 1, 6: 60}), ('Na@C60', {11: 1, 6: 60}), ('(Na@C60)+', {0: 1, 11: 1, 6: 60})])
@requires(parsing_library)
def test_formula_to_composition_caged(species, composition):
    """Should parse cage species."""
    assert formula_to_composition(species) == composition