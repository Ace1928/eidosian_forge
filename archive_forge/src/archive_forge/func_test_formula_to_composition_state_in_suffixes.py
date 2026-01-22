import pytest
from pyparsing import ParseException
from ..parsing import (
from ..testing import requires
@pytest.mark.parametrize('species, composition', [('CO2(g)', {6: 1, 8: 2}), ('CO2(l)', {6: 1, 8: 2}), ('CO2(s)', {6: 1, 8: 2})])
@requires(parsing_library)
def test_formula_to_composition_state_in_suffixes(species, composition):
    """Should parse species with state in suffixes."""
    assert formula_to_composition(species, suffixes=('(g)', '(l)', '(s)')) == composition