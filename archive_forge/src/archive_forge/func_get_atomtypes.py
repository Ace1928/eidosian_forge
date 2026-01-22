import re
import numpy as np
from ase import Atoms
from ase.utils import reader, writer
from ase.io.utils import ImageIterator
from ase.io import ParseError
from .vasp_parsers import vasp_outcar_parsers as vop
from pathlib import Path
def get_atomtypes(fname):
    """Given a file name, get the atomic symbols.

    The function can get this information from OUTCAR and POTCAR
    format files.  The files can also be compressed with gzip or
    bzip2.

    """
    fpath = Path(fname)
    atomtypes = []
    atomtypes_alt = []
    if fpath.suffix == '.gz':
        import gzip
        opener = gzip.open
    elif fpath.suffix == '.bz2':
        import bz2
        opener = bz2.BZ2File
    else:
        opener = open
    with opener(fpath) as fd:
        for line in fd:
            if 'TITEL' in line:
                atomtypes.append(line.split()[3].split('_')[0].split('.')[0])
            elif 'POTCAR:' in line:
                atomtypes_alt.append(line.split()[2].split('_')[0].split('.')[0])
    if len(atomtypes) == 0 and len(atomtypes_alt) > 0:
        if len(atomtypes_alt) % 2 != 0:
            raise ParseError(f'Tried to get atom types from {len(atomtypes_alt)} "POTCAR": lines in OUTCAR, but expected an even number')
        atomtypes = atomtypes_alt[0:len(atomtypes_alt) // 2]
    return atomtypes