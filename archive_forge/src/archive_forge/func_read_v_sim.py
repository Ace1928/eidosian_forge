import numpy as np
from ase.utils import reader, writer
@reader
def read_v_sim(fd):
    """Import V_Sim input file.

    Reads cell, atom positions, etc. from v_sim ascii file
    """
    from ase import Atoms, units
    from ase.geometry import cellpar_to_cell
    import re
    fd.readline()
    line = fd.readline() + ' ' + fd.readline()
    box = line.split()
    for i in range(len(box)):
        box[i] = float(box[i])
    keywords = []
    positions = []
    symbols = []
    unit = 1.0
    re_comment = re.compile('^\\s*[#!]')
    re_node = re.compile('^\\s*\\S+\\s+\\S+\\s+\\S+\\s+\\S+')
    while True:
        line = fd.readline()
        if line == '':
            break
        p = re_comment.match(line)
        if p is not None:
            line = line[p.end():].replace(',', ' ').lower()
            if line[:8] == 'keyword:':
                keywords.extend(line[8:].split())
        elif re_node.match(line):
            unit = 1.0
            if not 'reduced' in keywords:
                if 'bohr' in keywords or 'bohrd0' in keywords or 'atomic' in keywords or ('atomicd0' in keywords):
                    unit = units.Bohr
            fields = line.split()
            positions.append([unit * float(fields[0]), unit * float(fields[1]), unit * float(fields[2])])
            symbols.append(fields[3])
    if 'surface' in keywords or 'freeBC' in keywords:
        raise NotImplementedError
    if 'angdeg' in keywords:
        cell = cellpar_to_cell(box)
    else:
        unit = 1.0
        if 'bohr' in keywords or 'bohrd0' in keywords or 'atomic' in keywords or ('atomicd0' in keywords):
            unit = units.Bohr
        cell = np.zeros((3, 3))
        cell.flat[[0, 3, 4, 6, 7, 8]] = box[:6]
        cell *= unit
    if 'reduced' in keywords:
        atoms = Atoms(cell=cell, scaled_positions=positions)
    else:
        atoms = Atoms(cell=cell, positions=positions)
    atoms.set_chemical_symbols(symbols)
    return atoms