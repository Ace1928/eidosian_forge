import pytest
from pyparsing import ParseException
from ..parsing import (
from ..testing import requires
@pytest.mark.parametrize('species, unicode', [('Li@C60', 'Li@C₆₀'), ('(Li@C60)+', '(Li@C₆₀)⁺'), ('Na@C60', 'Na@C₆₀'), ('(Na@C60)+', '(Na@C₆₀)⁺')])
@requires(parsing_library)
def test_formula_to_unicode_caged(species, unicode):
    """Should produce LaTeX for cage species."""
    assert formula_to_unicode(species) == unicode