import pytest
from ase.build import molecule
from ase.calculators.emt import EMT
def test_set_calculator(atoms):
    calc = EMT()
    with pytest.deprecated_call():
        atoms.set_calculator(calc)
    assert atoms.calc is calc