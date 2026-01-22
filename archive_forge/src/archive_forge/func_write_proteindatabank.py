import warnings
import numpy as np
from ase.atoms import Atoms
from ase.geometry import cellpar_to_cell
from ase.io.espresso import label_to_symbol
from ase.utils import reader, writer
@writer
def write_proteindatabank(fileobj, images, write_arrays=True):
    """Write images to PDB-file."""
    if hasattr(images, 'get_positions'):
        images = [images]
    rotation = None
    if images[0].get_pbc().any():
        from ase.geometry import cell_to_cellpar, cellpar_to_cell
        currentcell = images[0].get_cell()
        cellpar = cell_to_cellpar(currentcell)
        exportedcell = cellpar_to_cell(cellpar)
        rotation = np.linalg.solve(currentcell, exportedcell)
        format = 'CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1\n'
        fileobj.write(format % (cellpar[0], cellpar[1], cellpar[2], cellpar[3], cellpar[4], cellpar[5]))
    format = 'ATOM  %5d %4s MOL     1    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s  \n'
    MAXNUM = 100000
    symbols = images[0].get_chemical_symbols()
    natoms = len(symbols)
    for n, atoms in enumerate(images):
        fileobj.write('MODEL     ' + str(n + 1) + '\n')
        p = atoms.get_positions()
        occupancy = np.ones(len(atoms))
        bfactor = np.zeros(len(atoms))
        if write_arrays:
            if 'occupancy' in atoms.arrays:
                occupancy = atoms.get_array('occupancy')
            if 'bfactor' in atoms.arrays:
                bfactor = atoms.get_array('bfactor')
        if rotation is not None:
            p = p.dot(rotation)
        for a in range(natoms):
            x, y, z = p[a]
            occ = occupancy[a]
            bf = bfactor[a]
            fileobj.write(format % ((a + 1) % MAXNUM, symbols[a], x, y, z, occ, bf, symbols[a].upper()))
        fileobj.write('ENDMDL\n')