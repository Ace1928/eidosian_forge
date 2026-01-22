import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.utils import reader, writer
@reader
def read_gromos(fileobj):
    """Read gromos geometry files (.g96).
    Reads:
    atom positions,
    and simulation cell (if present)
    tries to set atom types
    """
    lines = fileobj.readlines()
    read_pos = False
    read_box = False
    tmp_pos = []
    symbols = []
    mycell = None
    for line in lines:
        if read_pos and 'END' in line:
            read_pos = False
        if read_box and 'END' in line:
            read_box = False
        if read_pos:
            symbol, dummy, x, y, z = line.split()[2:7]
            tmp_pos.append((10 * float(x), 10 * float(y), 10 * float(z)))
            if len(symbol) != 2:
                symbols.append(symbol[0].lower().capitalize())
            else:
                symbol2 = symbol[0].lower().capitalize() + symbol[1]
                if symbol2 in chemical_symbols:
                    symbols.append(symbol2)
                else:
                    symbols.append(symbol[0].lower().capitalize())
            if symbols[-1] not in chemical_symbols:
                raise RuntimeError("Symbol '{}' not in chemical symbols".format(symbols[-1]))
        if read_box:
            try:
                grocell = list(map(float, line.split()))
            except ValueError:
                pass
            else:
                mycell = np.diag(grocell[:3])
                if len(grocell) >= 9:
                    mycell.flat[[1, 2, 3, 5, 6, 7]] = grocell[3:9]
                mycell *= 10.0
        if 'POSITION' in line:
            read_pos = True
        if 'BOX' in line:
            read_box = True
    gmx_system = Atoms(symbols=symbols, positions=tmp_pos, cell=mycell)
    if mycell is not None:
        gmx_system.pbc = True
    return gmx_system