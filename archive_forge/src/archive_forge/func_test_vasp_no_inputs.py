import pytest
from ase.build import bulk
@calc('vasp')
def test_vasp_no_inputs(system, factory):
    calc = factory.calc()
    system.calc = calc
    system.get_potential_energy()
    calc.read_incar('INCAR')
    assert calc.float_params['nelect'] is None