import numpy as np
import time
from ase.atoms import Atoms
from ase.io import read
from ase.units import Bohr
def write_cube(fileobj, atoms, data=None, origin=None, comment=None):
    """
    Function to write a cube file.

    fileobj: str or file object
        File to which output is written.
    atoms: Atoms object
        Atoms object specifying the atomic configuration.
    data : 3dim numpy array, optional (default = None)
        Array containing volumetric data as e.g. electronic density
    origin : 3-tuple
        Origin of the volumetric data (units: Angstrom)
    comment : str, optional (default = None)
        Comment for the first line of the cube file.
    """
    if data is None:
        data = np.ones((2, 2, 2))
    data = np.asarray(data)
    if data.dtype == complex:
        data = np.abs(data)
    if comment is None:
        comment = 'Cube file from ASE, written on ' + time.strftime('%c')
    else:
        comment = comment.strip()
    fileobj.write(comment)
    fileobj.write('\nOUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n')
    if origin is None:
        origin = np.zeros(3)
    else:
        origin = np.asarray(origin) / Bohr
    fileobj.write('{0:5}{1:12.6f}{2:12.6f}{3:12.6f}\n'.format(len(atoms), *origin))
    for i in range(3):
        n = data.shape[i]
        d = atoms.cell[i] / n / Bohr
        fileobj.write('{0:5}{1:12.6f}{2:12.6f}{3:12.6f}\n'.format(n, *d))
    positions = atoms.positions / Bohr
    numbers = atoms.numbers
    for Z, (x, y, z) in zip(numbers, positions):
        fileobj.write('{0:5}{1:12.6f}{2:12.6f}{3:12.6f}{4:12.6f}\n'.format(Z, 0.0, x, y, z))
    data.tofile(fileobj, sep='\n', format='%e')