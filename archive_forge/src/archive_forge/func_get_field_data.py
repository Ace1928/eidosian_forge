import string
import numpy as np
from ase.io import string2index
from ase.io.formats import parse_filename
from ase.data import chemical_symbols
def get_field_data(atoms1, atoms2, field):
    if field[0] == 'r':
        field = field[1:]
        rank_order = True
    else:
        rank_order = False
    if field in atoms_props:
        if field == 't':
            data = atoms1.get_tags()
        elif field == 'an':
            data = atoms1.numbers
        elif field == 'el':
            data = np.array([sym2num[sym] for sym in atoms1.symbols])
        elif field == 'i':
            data = np.arange(len(atoms1))
        else:
            if field.startswith('d'):
                y = atoms2.positions - atoms1.positions
            elif field.startswith('p'):
                if field[1] == '1':
                    y = atoms1.positions
                else:
                    y = atoms2.positions
            if field.endswith('x'):
                data = y[:, 0]
            elif field.endswith('y'):
                data = y[:, 1]
            elif field.endswith('z'):
                data = y[:, 2]
            else:
                data = np.linalg.norm(y, axis=1)
    else:
        if field[0] == 'd':
            y = atoms2.get_forces() - atoms1.get_forces()
        elif field[0] == 'a':
            y = (atoms2.get_forces() + atoms1.get_forces()) / 2
        elif field[1] == '1':
            y = atoms1.get_forces()
        else:
            y = atoms2.get_forces()
        if field.endswith('x'):
            data = y[:, 0]
        elif field.endswith('y'):
            data = y[:, 1]
        elif field.endswith('z'):
            data = y[:, 2]
        else:
            data = np.linalg.norm(y, axis=1)
    if rank_order:
        return np.argsort(np.argsort(-data))
    return data