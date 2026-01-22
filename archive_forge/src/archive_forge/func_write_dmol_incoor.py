from datetime import datetime
import numpy as np
from ase import Atom, Atoms
from ase.geometry.cell import cell_to_cellpar, cellpar_to_cell
from ase.units import Bohr
from ase.utils import reader, writer
@writer
def write_dmol_incoor(fd, atoms, bohr=True):
    """ Write a dmol incoor-file from an Atoms object

    Notes
    -----
    Only used for pbc 111.
    Can not handle multiple images.
    DMol3 expect data in .incoor files to be in bohr, if bohr is false however
    the data is written in Angstroms.
    """
    if not np.all(atoms.pbc):
        raise ValueError('PBC must be all true for .incoor format')
    if bohr:
        cell = atoms.cell / Bohr
        positions = atoms.positions / Bohr
    else:
        cell = atoms.cell
        positions = atoms.positions
    fd.write('$cell vectors\n')
    fd.write('            %18.14f  %18.14f  %18.14f\n' % (cell[0, 0], cell[0, 1], cell[0, 2]))
    fd.write('            %18.14f  %18.14f  %18.14f\n' % (cell[1, 0], cell[1, 1], cell[1, 2]))
    fd.write('            %18.14f  %18.14f  %18.14f\n' % (cell[2, 0], cell[2, 1], cell[2, 2]))
    fd.write('$coordinates\n')
    for a, pos in zip(atoms, positions):
        fd.write('%-12s%18.14f  %18.14f  %18.14f \n' % (a.symbol, pos[0], pos[1], pos[2]))
    fd.write('$end\n')