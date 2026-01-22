import pytest
from pyparsing import ParseException
from ..parsing import (
from ..testing import requires
@pytest.mark.parametrize('species, composition', [('e-', {0: -1}), ('e-1', {0: -1}), ('e-(aq)', {0: -1})])
@requires(parsing_library)
def test_formula_to_composition_solvated_electrons(species, composition):
    assert formula_to_composition(species) == composition