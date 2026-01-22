import numpy as np
import time
from ase.atoms import Atoms
from ase.io import read
from ase.units import Bohr
def read_cube(fileobj, read_data=True, program=None, verbose=False):
    """Read atoms and data from CUBE file.

    fileobj : str or file
        Location to the cubefile.
    read_data : boolean
        If set true, the actual cube file content, i.e. an array
        containing the electronic density (or something else )on a grid
        and the dimensions of the corresponding voxels are read.
    program: str
        Use program='castep' to follow the PBC convention that first and
        last voxel along a direction are mirror images, thus the last
        voxel is to be removed.  If program=None, the routine will try
        to catch castep files from the comment lines.
    verbose : bool
        Print some more information to stdout.

    Returns a dict with the following keys:
    
    * 'atoms': Atoms object
    * 'data' : (Nx, Ny, Nz) ndarray
    * 'origin': (3,) ndarray, specifying the cube_data origin.
    """
    readline = fileobj.readline
    line = readline()
    line = readline()
    axes = []
    if 'OUTER LOOP' in line.upper():
        axes = ['XYZ'.index(s[0]) for s in line.upper().split()[2::3]]
    if not axes:
        axes = [0, 1, 2]
    if 'castep2cube' in line:
        program = 'castep'
        if verbose:
            print('read_cube identified program: castep')
    line = readline().split()
    natoms = int(line[0])
    origin = np.array([float(x) * Bohr for x in line[1:]])
    cell = np.empty((3, 3))
    shape = []
    for i in range(3):
        n, x, y, z = [float(s) for s in readline().split()]
        shape.append(int(n))
        if program == 'castep':
            n -= 1
        cell[i] = n * Bohr * np.array([x, y, z])
    numbers = np.empty(natoms, int)
    positions = np.empty((natoms, 3))
    for i in range(natoms):
        line = readline().split()
        numbers[i] = int(line[0])
        positions[i] = [float(s) for s in line[2:]]
    positions *= Bohr
    atoms = Atoms(numbers=numbers, positions=positions, cell=cell)
    if program == 'castep':
        atoms.pbc = True
    dct = {'atoms': atoms}
    if read_data:
        data = np.array([float(s) for s in fileobj.read().split()]).reshape(shape)
        if axes != [0, 1, 2]:
            data = data.transpose(axes).copy()
        if program == 'castep':
            data = data[:-1, :-1, :-1]
        dct['data'] = data
        dct['origin'] = origin
    return dct