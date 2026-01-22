import pytest
from ..periodic import (
from ..testing import requires
from ..parsing import formula_to_composition, parsing_library
def test_relative_atomic_masses():
    assert relative_atomic_masses[0] == 1.008