from ase.atoms import Atoms
from ase.units import Bohr
from re import compile
def read_sys(fileobj):
    """
    Function to read a qb@ll sys file.

    fileobj: file (object) to read from.
    """
    a1, a2, a3, b1, b2, b3, c1, c2, c3 = fileobj.readline().split()[2:11]
    cell = []
    cell.append([float(a1) * Bohr, float(a2) * Bohr, float(a3) * Bohr])
    cell.append([float(b1) * Bohr, float(b2) * Bohr, float(b3) * Bohr])
    cell.append([float(c1) * Bohr, float(c2) * Bohr, float(c3) * Bohr])
    positions = []
    symbols = []
    reg = compile('(\\d+|\\s+)')
    line = fileobj.readline()
    while 'species' in line:
        line = fileobj.readline()
    while line:
        a, symlabel, spec, x, y, z = line.split()[0:6]
        positions.append([float(x) * Bohr, float(y) * Bohr, float(z) * Bohr])
        sym = reg.split(str(symlabel))
        symbols.append(sym[0])
        line = fileobj.readline()
    atoms = Atoms(symbols=symbols, cell=cell, positions=positions)
    return atoms