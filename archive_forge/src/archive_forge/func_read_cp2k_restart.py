import numpy as np
from itertools import islice
import os
from ase.atoms import Atoms, Atom
from ase.cell import Cell
from ase.io.formats import index2range
from ase.data import atomic_numbers
def read_cp2k_restart(fileobj):
    """Read Atoms and Cell from Restart File.

    Reads the elements, coordinates and cell information from the
    '&SUBSYS' section of a CP2K restart file.

    Tries to convert atom names to elements, if this fails element is set to X.

    Returns an Atoms object.
    """

    def _parse_section(inp):
        """Helper to parse structure to nested dict"""
        ret = {'content': []}
        while inp:
            line = inp.readline().strip()
            if line.startswith('&END'):
                return ret
            elif line.startswith('&'):
                key = line.replace('&', '')
                ret[key] = _parse_section(inp)
            else:
                ret['content'].append(line)
        return ret

    def _fast_forward_to(fileobj, section_header):
        """Helper to forward to a section"""
        found = False
        while fileobj:
            line = fileobj.readline()
            if section_header in line:
                found = True
                break
        if not found:
            raise RuntimeError('No {:} section found!'.format(section_header))

    def _read_cell(data):
        """Helper to read cell data, returns cell and pbc"""
        cell = None
        pbc = [False, False, False]
        if 'CELL' in data:
            content = data['CELL']['content']
            cell = Cell([[0, 0, 0] for i in range(3)])
            char2idx = {'A ': 0, 'B ': 1, 'C ': 2}
            for line in content:
                if line[:2] in char2idx:
                    idx = char2idx[line[:2]]
                    cell[idx] = [float(x) for x in line.split()[1:]]
                    pbc[idx] = True
            if not set([len(v) for v in cell]) == {3}:
                raise RuntimeError('Bad Cell Definition found.')
        return (cell, pbc)

    def _read_geometry(content):
        """Helper to read geometry, returns a list of Atoms"""
        atom_list = []
        for entry in content:
            entry = entry.split()
            el = [char.lower() for char in entry[0] if char.isalpha()]
            el = ''.join(el).capitalize()
            pos = [float(x) for x in entry[1:4]]
            if el in atomic_numbers.keys():
                atom_list.append(Atom(el, pos))
            else:
                atom_list.append(Atom('X', pos))
        return atom_list
    _fast_forward_to(fileobj, '&SUBSYS')
    data = _parse_section(fileobj)
    cell, pbc = _read_cell(data)
    atom_list = _read_geometry(data['COORD']['content'])
    return Atoms(atom_list, cell=cell, pbc=pbc)