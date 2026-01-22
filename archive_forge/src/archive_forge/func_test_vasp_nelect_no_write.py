import pytest
from ase.build import bulk
@calc('vasp')
def test_vasp_nelect_no_write(factory, system):
    calc = factory.calc(xc='LDA', nsw=-1, ibrion=-1, nelm=1, lwave=False, lcharg=False, charge=0)
    calc.initialize(system)
    calc.write_input(system)
    calc.read_incar('INCAR')
    assert calc.float_params['nelect'] is None