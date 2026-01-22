import time
import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.calculators.lammpsrun import Prism
from ase.neighborlist import NeighborList
from ase.data import atomic_masses, chemical_symbols
from ase.io import read
def read_connectivities(self, fileobj, update_types=False):
    """Read positions, connectivities, etc.

        update_types: update atom types from the masses
        """
    lines = fileobj.readlines()
    lines.pop(0)

    def next_entry():
        line = lines.pop(0).strip()
        if len(line) > 0:
            lines.insert(0, line)

    def next_key():
        while len(lines):
            line = lines.pop(0).strip()
            if len(line) > 0:
                lines.pop(0)
                return line
        return None
    next_entry()
    header = {}
    while True:
        line = lines.pop(0).strip()
        if len(line):
            w = line.split()
            if len(w) == 2:
                header[w[1]] = int(w[0])
            else:
                header[w[1] + ' ' + w[2]] = int(w[0])
        else:
            break
    while not lines.pop(0).startswith('Atoms'):
        pass
    lines.pop(0)
    natoms = len(self)
    positions = np.empty((natoms, 3))
    for i in range(natoms):
        w = lines.pop(0).split()
        assert int(w[0]) == i + 1
        positions[i] = np.array([float(w[4 + c]) for c in range(3)])
    key = next_key()
    velocities = None
    if key == 'Velocities':
        velocities = np.empty((natoms, 3))
        for i in range(natoms):
            w = lines.pop(0).split()
            assert int(w[0]) == i + 1
            velocities[i] = np.array([float(w[1 + c]) for c in range(3)])
        key = next_key()
    if key == 'Masses':
        ntypes = len(self.types)
        masses = np.empty(ntypes)
        for i in range(ntypes):
            w = lines.pop(0).split()
            assert int(w[0]) == i + 1
            masses[i] = float(w[1])
        if update_types:

            def newtype(element, types):
                if len(element) > 1:
                    return element
                count = 0
                for type in types:
                    if type[0] == element:
                        count += 1
                label = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                return element + label[count]
            symbolmap = {}
            typemap = {}
            types = []
            ams = atomic_masses[:]
            ams[np.isnan(ams)] = 0
            for i, mass in enumerate(masses):
                m2 = (ams - mass) ** 2
                symbolmap[self.types[i]] = chemical_symbols[m2.argmin()]
                typemap[self.types[i]] = newtype(chemical_symbols[m2.argmin()], types)
                types.append(typemap[self.types[i]])
            for atom in self:
                atom.symbol = symbolmap[atom.symbol]
            self.types = types
        key = next_key()

    def read_list(key_string, length, debug=False):
        if key != key_string:
            return ([], key)
        lst = []
        while len(lines):
            w = lines.pop(0).split()
            if len(w) > length:
                lst.append([int(w[1 + c]) - 1 for c in range(length)])
            else:
                return (lst, next_key())
        return (lst, None)
    bonds, key = read_list('Bonds', 3)
    angles, key = read_list('Angles', 4)
    dihedrals, key = read_list('Dihedrals', 5, True)
    self.connectivities = {'bonds': bonds, 'angles': angles, 'dihedrals': dihedrals}
    if 'bonds' in header:
        assert len(bonds) == header['bonds']
        self.connectivities['bond types'] = list(range(header['bond types']))
    if 'angles' in header:
        assert len(angles) == header['angles']
        self.connectivities['angle types'] = list(range(header['angle types']))
    if 'dihedrals' in header:
        assert len(dihedrals) == header['dihedrals']
        self.connectivities['dihedral types'] = list(range(header['dihedral types']))