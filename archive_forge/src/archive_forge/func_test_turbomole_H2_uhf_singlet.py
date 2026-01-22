from ase import Atoms
from ase.calculators.turbomole import Turbomole
import os.path
import pytest
def test_turbomole_H2_uhf_singlet(atoms):
    atoms.calc = Turbomole(**{'multiplicity': 1, 'uhf': True, 'use dft': True})
    atoms.get_potential_energy()
    dft_in_output = False
    with open('ASE.TM.dscf.out') as fd:
        for line in fd:
            if 'density functional' in line:
                dft_in_output = True
    assert dft_in_output
    assert os.path.exists('alpha')
    assert os.path.exists('beta')
    assert not os.path.exists('mos')