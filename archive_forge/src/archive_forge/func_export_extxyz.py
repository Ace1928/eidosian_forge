import numpy as np
from ase.calculators.calculator import Calculator
from ase.data import atomic_numbers
from ase.utils import IOContext
from ase.geometry import get_distances
from ase.cell import Cell
def export_extxyz(self, atoms=None, filename='qmmm_atoms.xyz'):
    """
        exports the atoms to extended xyz file with additional "region"
        array keeping the mapping between QM, buffer and MM parts of
        the simulation
        """
    if atoms is None:
        if self.atoms is None:
            raise ValueError('Calculator has no atoms')
        else:
            atoms = self.atoms
    region = self.get_region_from_masks(atoms=atoms)
    atoms_copy = atoms.copy()
    atoms_copy.new_array('region', region)
    atoms_copy.calc = self
    atoms_copy.write(filename, format='extxyz')