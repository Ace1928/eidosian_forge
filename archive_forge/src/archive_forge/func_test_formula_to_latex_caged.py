import pytest
from pyparsing import ParseException
from ..parsing import (
from ..testing import requires
@pytest.mark.parametrize('species, latex', [('Li@C60', 'Li@C_{60}'), ('(Li@C60)+', '(Li@C_{60})^{+}'), ('Na@C60', 'Na@C_{60}'), ('(Na@C60)+', '(Na@C_{60})^{+}')])
@requires(parsing_library)
def test_formula_to_latex_caged(species, latex):
    """Should produce LaTeX for cage species."""
    assert formula_to_latex(species) == latex