import re
import numpy as np
from ase import Atoms
from ase.utils import reader, writer
from ase.io.utils import ImageIterator
from ase.io import ParseError
from .vasp_parsers import vasp_outcar_parsers as vop
from pathlib import Path
@reader
def read_vasp_xdatcar(filename='XDATCAR', index=-1):
    """Import XDATCAR file

       Reads all positions from the XDATCAR and returns a list of
       Atoms objects.  Useful for viewing optimizations runs
       from VASP5.x

       Constraints ARE NOT stored in the XDATCAR, and as such, Atoms
       objects retrieved from the XDATCAR will not have constraints set.
    """
    fd = filename
    images = list()
    cell = np.eye(3)
    atomic_formula = str()
    while True:
        comment_line = fd.readline()
        if 'Direct configuration=' not in comment_line:
            try:
                lattice_constant = float(fd.readline())
            except Exception:
                break
            xx = [float(x) for x in fd.readline().split()]
            yy = [float(y) for y in fd.readline().split()]
            zz = [float(z) for z in fd.readline().split()]
            cell = np.array([xx, yy, zz]) * lattice_constant
            symbols = fd.readline().split()
            numbers = [int(n) for n in fd.readline().split()]
            total = sum(numbers)
            atomic_formula = ''.join(('{:s}{:d}'.format(sym, numbers[n]) for n, sym in enumerate(symbols)))
            fd.readline()
        coords = [np.array(fd.readline().split(), float) for ii in range(total)]
        image = Atoms(atomic_formula, cell=cell, pbc=True)
        image.set_scaled_positions(np.array(coords))
        images.append(image)
    if not index:
        return images
    else:
        return images[index]