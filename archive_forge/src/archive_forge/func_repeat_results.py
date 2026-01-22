from math import sqrt
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.data import covalent_radii
from ase.gui.defaults import read_defaults
from ase.io import read, write, string2index
from ase.gui.i18n import _
from ase.geometry import find_mic
import warnings
def repeat_results(self, atoms, repeat=None, oldprod=None):
    """Return a dictionary which updates the magmoms, energy and forces
        to the repeated amount of atoms.
        """

    def getresult(name, get_quantity):
        try:
            if not atoms.calc or atoms.calc.calculation_required(atoms, [name]):
                quantity = None
            else:
                quantity = get_quantity()
        except Exception as err:
            quantity = None
            errmsg = 'An error occurred while retrieving {} from the calculator: {}'.format(name, err)
            warnings.warn(errmsg)
        return quantity
    if repeat is None:
        repeat = self.repeat.prod()
    if oldprod is None:
        oldprod = self.repeat.prod()
    results = {}
    original_length = len(atoms) // oldprod
    newprod = repeat.prod()
    magmoms = getresult('magmoms', atoms.get_magnetic_moments)
    magmom = getresult('magmom', atoms.get_magnetic_moment)
    energy = getresult('energy', atoms.get_potential_energy)
    forces = getresult('forces', atoms.get_forces)
    if magmoms is not None:
        magmoms = np.tile(magmoms[:original_length], newprod)
        results['magmoms'] = magmoms
    if magmom is not None:
        magmom = magmom * newprod / oldprod
        results['magmom'] = magmom
    if forces is not None:
        forces = np.tile(forces[:original_length].T, newprod).T
        results['forces'] = forces
    if energy is not None:
        energy = energy * newprod / oldprod
        results['energy'] = energy
    return results