import pytest
from ..periodic import (
from ..testing import requires
from ..parsing import formula_to_composition, parsing_library
def test_atomic_number():
    assert atomic_number('U') == 92
    assert atomic_number('u') == 92
    assert atomic_number('carbon') == 6
    assert atomic_number('oganesson') == 118
    with pytest.raises(ValueError):
        atomic_number('unobtainium')