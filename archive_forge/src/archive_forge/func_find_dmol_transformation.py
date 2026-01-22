import os
import re
import numpy as np
from ase import Atoms
from ase.io import read
from ase.io.dmol import write_dmol_car, write_dmol_incoor
from ase.units import Hartree, Bohr
from ase.calculators.calculator import FileIOCalculator, Parameters, ReadError
def find_dmol_transformation(self, tol=0.0001):
    """Finds rotation matrix that takes us from DMol internal
        coordinates to ase coordinates.

        For pbc = [False, False, False]  the rotation matrix is parsed from
        the .rot file, if this file doesnt exist no rotation is needed.

        For pbc = [True, True, True] the Dmol internal cell vectors and
        positions are parsed and compared to self.ase_cell self.ase_positions.
        The rotation matrix can then be found by a call to the helper
        function find_transformation(atoms1, atoms2)

        If a rotation matrix is needed then self.internal_transformation is
        set to True and the rotation matrix is stored in self.rotation_matrix

        Parameters
        ----------
        tol (float): tolerance for check if positions and cell are the same
        """
    if np.all(self.atoms.pbc):
        dmol_atoms = self.read_atoms_from_outmol()
        if np.linalg.norm(self.atoms.positions - dmol_atoms.positions) < tol and np.linalg.norm(self.atoms.cell - dmol_atoms.cell) < tol:
            self.internal_transformation = False
        else:
            R, err = find_transformation(dmol_atoms, self.atoms)
            if abs(np.linalg.det(R) - 1.0) > tol:
                raise RuntimeError('Error: transformation matrix does not have determinant 1.0')
            if err < tol:
                self.internal_transformation = True
                self.rotation_matrix = R
            else:
                raise RuntimeError('Error: Could not find dmol coordinate transformation')
    elif not np.any(self.atoms.pbc):
        try:
            data = np.loadtxt(self.label + '.rot')
        except IOError:
            self.internal_transformation = False
        else:
            self.internal_transformation = True
            self.rotation_matrix = data[1:].transpose()