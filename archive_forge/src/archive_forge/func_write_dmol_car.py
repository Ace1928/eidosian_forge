from datetime import datetime
import numpy as np
from ase import Atom, Atoms
from ase.geometry.cell import cell_to_cellpar, cellpar_to_cell
from ase.units import Bohr
from ase.utils import reader, writer
@writer
def write_dmol_car(fd, atoms):
    """ Write a dmol car-file from an Atoms object

    Notes
    -----
    The positions written to file are rotated as to align with the cell when
    reading (due to cellpar information)
    Can not handle multiple images.
    Only allows for pbc 111 or 000.
    """
    fd.write('!BIOSYM archive 3\n')
    dt = datetime.now()
    symbols = atoms.get_chemical_symbols()
    if np.all(atoms.pbc):
        cellpar = cell_to_cellpar(atoms.cell)
        new_cell = cellpar_to_cell(cellpar)
        lstsq_fit = np.linalg.lstsq(atoms.cell, new_cell, rcond=-1)
        R = lstsq_fit[0]
        positions = np.dot(atoms.positions, R)
        fd.write('PBC=ON\n\n')
        fd.write('!DATE     %s\n' % dt.strftime('%b %d %H:%m:%S %Y'))
        fd.write('PBC %9.5f %9.5f %9.5f %9.5f %9.5f %9.5f\n' % tuple(cellpar))
    elif not np.any(atoms.pbc):
        fd.write('PBC=OFF\n\n')
        fd.write('!DATE     %s\n' % dt.strftime('%b %d %H:%m:%S %Y'))
        positions = atoms.positions
    else:
        raise ValueError('PBC must be all true or all false for .car format')
    for i, (sym, pos) in enumerate(zip(symbols, positions)):
        fd.write('%-6s  %12.8f   %12.8f   %12.8f XXXX 1      xx      %-2s  0.000\n' % (sym + str(i + 1), pos[0], pos[1], pos[2], sym))
    fd.write('end\nend\n')