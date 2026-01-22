import pytest
from pyparsing import ParseException
from ..parsing import (
from ..testing import requires
@pytest.mark.parametrize('species', ['Na+Cl-'])
@requires(parsing_library)
def test_formula_to_composition_bad_charges(species):
    with pytest.raises(ValueError):
        formula_to_composition(species)