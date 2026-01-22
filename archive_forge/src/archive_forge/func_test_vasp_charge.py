import pytest
from ase.build import bulk
@calc('vasp')
def test_vasp_charge(factory, system, expected_nelect_from_vasp):
    """
    Run VASP tests to ensure that determining number of electrons from
    user-supplied charge works correctly.

    Test that the number of charge found matches the expected.
    """
    calc = factory.calc(xc='LDA', nsw=-1, ibrion=-1, nelm=1, lwave=False, lcharg=False)
    system.calc = calc
    system.get_potential_energy()
    default_nelect_from_vasp = calc.get_number_of_electrons()
    assert default_nelect_from_vasp == expected_nelect_from_vasp