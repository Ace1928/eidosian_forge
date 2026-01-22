import pytest
from pyparsing import ParseException
from ..parsing import (
from ..testing import requires
@pytest.mark.parametrize('species, composition', [('CH4(g)', {1: 4, 6: 1}), ('CH3CH3(g)', {1: 6, 6: 2}), ('C6H6(l)', {1: 6, 6: 6}), ('(CH)6(l)', {1: 6, 6: 6}), ('CHCHCHCHCHCH(l)', {1: 6, 6: 6})])
@requires(parsing_library)
def test_formula_to_composition_organic_compounds(species, composition):
    assert formula_to_composition(species) == composition