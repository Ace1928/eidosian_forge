from collections import defaultdict
import pytest
from ..util.testing import requires
from ..units import (
@requires(units_library)
def test_default_unit_in_registry():
    mol_per_m3 = default_unit_in_registry(3 * u.molar, SI_base_registry)
    assert magnitude(mol_per_m3) == 1
    assert mol_per_m3 == u.mole / u.metre ** 3
    assert default_unit_in_registry(3, SI_base_registry) == 1
    assert default_unit_in_registry(3.0, SI_base_registry) == 1