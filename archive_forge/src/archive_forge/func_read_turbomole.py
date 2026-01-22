from ase.units import Bohr
def read_turbomole(fd):
    """Method to read turbomole coord file

    coords in bohr, atom types in lowercase, format:
    $coord
    x y z atomtype
    x y z atomtype f
    $end
    Above 'f' means a fixed atom.
    """
    from ase import Atoms
    from ase.constraints import FixAtoms
    lines = fd.readlines()
    atoms_pos = []
    atom_symbols = []
    myconstraints = []
    for i, l in enumerate(lines):
        if l.strip().startswith('$coord'):
            start = i
            break
    for line in lines[start + 1:]:
        if line.startswith('$'):
            break
        else:
            x, y, z, symbolraw = line.split()[:4]
            symbolshort = symbolraw.strip()
            symbol = symbolshort[0].upper() + symbolshort[1:].lower()
            atom_symbols.append(symbol)
            atoms_pos.append([float(x) * Bohr, float(y) * Bohr, float(z) * Bohr])
            cols = line.split()
            if len(cols) == 5:
                fixedstr = line.split()[4].strip()
                if fixedstr == 'f':
                    myconstraints.append(True)
                else:
                    myconstraints.append(False)
            else:
                myconstraints.append(False)
    atom_symbols = [element if element != 'Q' else 'X' for element in atom_symbols]
    atoms = Atoms(positions=atoms_pos, symbols=atom_symbols, pbc=False)
    c = FixAtoms(mask=myconstraints)
    atoms.set_constraint(c)
    return atoms