import os
import re
import numpy as np
from ase import Atoms
from ase.io import read
from ase.io.dmol import write_dmol_car, write_dmol_incoor
from ase.units import Hartree, Bohr
from ase.calculators.calculator import FileIOCalculator, Parameters, ReadError
def find_transformation(atoms1, atoms2, verbose=False, only_cell=False):
    """ Solves Ax = B where A and B are cell and positions from atoms objects.

    Uses numpys least square solver to solve the problem Ax = B where A and
    B are cell vectors and positions for atoms1 and atoms2 respectively.

    Parameters
    ----------
    atoms1 (Atoms object): First atoms object (A)
    atoms2 (Atoms object): Second atoms object (B)
    verbose (bool): If True prints for each i A[i], B[i], Ax[i]
    only_cell (bool): If True only cell in used, otherwise cell and positions.

    Returns
    -------
    x (np.array((3,3))): Least square solution to Ax = B
    error (float): The error calculated as np.linalg.norm(Ax-b)

    """
    if only_cell:
        N = 3
    elif len(atoms1) != len(atoms2):
        raise RuntimeError('Atoms object must be of same length')
    else:
        N = len(atoms1) + 3
    A = np.zeros((N, 3))
    B = np.zeros((N, 3))
    A[0:3, :] = atoms1.cell
    B[0:3, :] = atoms2.cell
    if not only_cell:
        A[3:, :] = atoms1.positions
        B[3:, :] = atoms2.positions
    lstsq_fit = np.linalg.lstsq(A, B, rcond=-1)
    x = lstsq_fit[0]
    error = np.linalg.norm(np.dot(A, x) - B)
    if verbose:
        print('%17s %33s %35s %24s' % ('A', 'B', 'Ax', '|Ax-b|'))
        for a, b in zip(A, B):
            ax = np.dot(a, x)
            loss = np.linalg.norm(ax - b)
            print('(', end='')
            for a_i in a:
                print('%8.5f' % a_i, end='')
            print(')   (', end='')
            for b_i in b:
                print('%8.5f ' % b_i, end='')
            print(')   (', end='')
            for ax_i in ax:
                print('%8.5f ' % ax_i, end='')
            print(')   %8.5f' % loss)
    return (x, error)