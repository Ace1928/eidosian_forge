import pytest
from pyparsing import ParseException
from ..parsing import (
from ..testing import requires
@pytest.mark.parametrize('species', ['ch3oh', 'Ch3oh', 'Ch3OH', 'ch3OH(l)'])
@requires(parsing_library)
def test_formula_to_composition_fail(species):
    """Should raise an exception."""
    with pytest.raises(ParseException):
        formula_to_composition(species)