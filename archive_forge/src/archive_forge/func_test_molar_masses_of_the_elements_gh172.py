import pytest
from ..periodic import (
from ..testing import requires
from ..parsing import formula_to_composition, parsing_library
def test_molar_masses_of_the_elements_gh172():
    from chempy import Substance
    previous_mass = 0
    for symbol in symbols:
        this_mass = Substance.from_formula(symbol).mass
        if symbol in ('K', 'Ni', 'I', 'Pa', 'Np', 'Am'):
            assert this_mass < previous_mass
        elif symbol in ('Bk', 'Og'):
            assert this_mass == previous_mass
        else:
            assert this_mass > previous_mass
        previous_mass = this_mass
    assert Substance.from_formula('Hs').mass == 271