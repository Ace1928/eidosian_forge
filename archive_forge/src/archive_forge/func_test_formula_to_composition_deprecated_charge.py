import pytest
from pyparsing import ParseException
from ..parsing import (
from ..testing import requires
@pytest.mark.parametrize('species', ['Cl/-', 'Cl/-(aq)', 'Fe(SCN)2/+', 'Fe(SCN)2/+(aq)', 'Fe/3+', 'Fe/3+(aq)', 'Na/+', 'Na/+(aq)', 'e/-', 'e/-(aq)', '.NO3/2-'])
@requires(parsing_library)
def test_formula_to_composition_deprecated_charge(species):
    with pytest.raises(ValueError):
        formula_to_composition(species)