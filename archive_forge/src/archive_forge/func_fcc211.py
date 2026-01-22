from math import sqrt
from operator import itemgetter
import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.data import reference_states, atomic_numbers
from ase.lattice.cubic import FaceCenteredCubic
def fcc211(symbol, size, a=None, vacuum=None, orthogonal=True):
    """FCC(211) surface.

    Does not currently support special adsorption sites.

    Currently only implemented for *orthogonal=True* with size specified
    as (i, j, k), where i, j, and k are number of atoms in each direction.
    i must be divisible by 3 to accommodate the step width.
    """
    if not orthogonal:
        raise NotImplementedError('Only implemented for orthogonal unit cells.')
    if size[0] % 3 != 0:
        raise NotImplementedError('First dimension of size must be divisible by 3.')
    atoms = FaceCenteredCubic(symbol, directions=[[1, -1, -1], [0, 2, -2], [2, 1, 1]], miller=(None, None, (2, 1, 1)), latticeconstant=a, size=(1, 1, 1), pbc=True)
    z = (size[2] + 1) // 2
    atoms = atoms.repeat((size[0] // 3, size[1], z))
    if size[2] % 2:
        remove_list = [atom.index for atom in atoms if atom.z < atoms[1].z]
        del atoms[remove_list]
        dz = atoms[0].z
        atoms.translate((0.0, 0.0, -dz))
        atoms.cell[2][2] -= dz
    atoms.cell[2] = 0.0
    atoms.pbc[2] = False
    if vacuum:
        atoms.center(vacuum, axis=2)
    orders = [(atom.index, round(atom.x, 3), round(atom.y, 3), -round(atom.z, 3), atom.index) for atom in atoms]
    orders.sort(key=itemgetter(3, 1, 2))
    newatoms = atoms.copy()
    for index, order in enumerate(orders):
        newatoms[index].position = atoms[order[0]].position.copy()
    newatoms.info['adsorbate_info'] = {'sites': {}}
    return newatoms