import warnings
import numpy as np
from ase.atoms import Atoms
from ase.geometry import cellpar_to_cell
from ase.io.espresso import label_to_symbol
from ase.utils import reader, writer
def read_atom_line(line_full):
    """
    Read atom line from pdb format
    HETATM    1  H14 ORTE    0       6.301   0.693   1.919  1.00  0.00        H
    """
    line = line_full.rstrip('\n')
    type_atm = line[0:6]
    if type_atm == 'ATOM  ' or type_atm == 'HETATM':
        name = line[12:16].strip()
        altloc = line[16]
        resname = line[17:21]
        resseq = int(line[22:26].split()[0])
        try:
            coord = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=np.float64)
        except ValueError:
            raise ValueError('Invalid or missing coordinate(s)')
        try:
            occupancy = float(line[54:60])
        except ValueError:
            occupancy = None
        if occupancy is not None and occupancy < 0:
            warnings.warn('Negative occupancy in one or more atoms')
        try:
            bfactor = float(line[60:66])
        except ValueError:
            bfactor = 0.0
        symbol = line[76:78].strip().upper()
    else:
        raise ValueError('Only ATOM and HETATM supported')
    return (symbol, name, altloc, resname, coord, occupancy, bfactor, resseq)