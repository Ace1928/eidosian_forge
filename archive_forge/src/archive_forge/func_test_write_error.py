import numpy as np
import pytest
from ase import Atoms
from ase.io import read
from ase.build import bulk
from .test_mustem import make_STO_atoms
def test_write_error():
    """Check missing parameter when writing xyz prismatic file."""
    atoms_Si100 = bulk('Si', cubic=True)
    atoms_STO = make_STO_atoms()
    filename = 'SI100.XYZ'
    with pytest.raises(ValueError):
        atoms_Si100.write(filename, format='prismatic')
    atoms_Si100.write(filename, format='prismatic', debye_waller_factors=0.076)
    atoms_Si100.write(filename, format='prismatic', debye_waller_factors={'Si': 0.076})
    STO_DW_dict = {'Sr': 0.62, 'O': 0.73, 'Ti': 0.43}
    STO_DW_dict_Ti_missing = {key: STO_DW_dict[key] for key in ['Sr', 'O']}
    with pytest.raises(ValueError):
        atoms_STO.write(filename, format='prismatic', debye_waller_factors=STO_DW_dict_Ti_missing)
    atoms_STO.write(filename, format='prismatic', debye_waller_factors=STO_DW_dict)
    with pytest.raises(ValueError):
        atoms4 = Atoms(['Sr', 'Ti', 'O', 'O', 'O'], positions=[[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
        atoms4.write(filename, format='prismatic', debye_waller_factors=STO_DW_dict)