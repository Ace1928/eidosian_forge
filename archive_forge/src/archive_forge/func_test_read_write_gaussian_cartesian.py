import copy
from io import StringIO
import numpy as np
import pytest
from ase.atoms import Atoms
from ase.calculators.calculator import InputError
from ase.calculators.gaussian import Gaussian
from ase.io import ParseError
from ase.io.gaussian import (_get_atoms_info, _get_cartesian_atom_coords,
def test_read_write_gaussian_cartesian(fd_cartesian, cartesian_setup):
    """Tests the read_gaussian_in and write_gaussian_in methods.
    For the input text given by each fixture we do the following:
    - Check reading in the text generates the Atoms object and Calculator that
      we would expect to get.
    - Check that writing out the resulting Atoms object and reading it back in
      generates the same Atoms object and parameters. """
    atoms, params = cartesian_setup
    params['nmagmlist'] = np.array([None, -8.89, None])
    params['zefflist'] = np.array([None, -1, None])
    params['znuclist'] = np.array([None, None, 2])
    params['qmomlist'] = np.array([None, None, 1])
    params['radnuclearlist'] = np.array([None, None, 1])
    params['spinlist'] = np.array([None, None, 1])
    with pytest.warns(UserWarning):
        atoms_new = read_gaussian_in(fd_cartesian, True)
    atoms_new.set_masses(_get_iso_masses(atoms_new))
    _check_atom_properties(atoms, atoms_new, params)
    _test_write_gaussian(atoms_new, params)