import os
import re
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import read
from ase.calculators.calculator import kpts2ndarray
from ase.units import Bohr, Hartree
from ase.utils import reader
def kwargs2atoms(kwargs, directory=None):
    """Extract atoms object from keywords and return remaining keywords.

    Some keyword arguments may refer to files.  The directory keyword
    may be necessary to resolve the paths correctly, and is used for
    example when running 'ase gui somedir/inp'."""
    kwargs = normalize_keywords(kwargs)
    coord_keywords = ['coordinates', 'xyzcoordinates', 'pdbcoordinates', 'reducedcoordinates', 'xsfcoordinates', 'xsfcoordinatesanimstep']
    nkeywords = 0
    for keyword in coord_keywords:
        if keyword in kwargs:
            nkeywords += 1
    if nkeywords == 0:
        raise OctopusParseError('No coordinates')
    elif nkeywords > 1:
        raise OctopusParseError('Multiple coordinate specifications present.  This may be okay in Octopus, but we do not implement it.')

    def get_positions_from_block(keyword):
        block = kwargs.pop(keyword)
        positions = []
        numbers = []
        tags = []
        types = {}
        for row in block:
            assert len(row) in [ndims + 1, ndims + 2]
            row = row[:ndims + 1]
            sym = row[0]
            assert sym.startswith('"') or sym.startswith("'")
            assert sym[0] == sym[-1]
            sym = sym[1:-1]
            pos0 = np.zeros(3)
            ndim = int(kwargs.get('dimensions', 3))
            pos0[:ndim] = [float(element) for element in row[1:]]
            number = atomic_numbers.get(sym)
            tag = 0
            if number is None:
                if sym not in types:
                    tag = len(types) + 1
                    types[sym] = tag
                number = 0
                tag = types[sym]
            tags.append(tag)
            numbers.append(number)
            positions.append(pos0)
        positions = np.array(positions)
        tags = np.array(tags, int)
        if types:
            ase_types = {}
            for sym, tag in types.items():
                ase_types['X', tag] = sym
            info = {'types': ase_types}
        else:
            tags = None
            info = None
        return (numbers, positions, tags, info)

    def read_atoms_from_file(fname, fmt):
        assert fname.startswith('"') or fname.startswith("'")
        assert fname[0] == fname[-1]
        fname = fname[1:-1]
        if directory is not None:
            fname = os.path.join(directory, fname)
        if fmt == 'xsf' and 'xsfcoordinatesanimstep' in kwargs:
            anim_step = kwargs.pop('xsfcoordinatesanimstep')
            theslice = slice(anim_step, anim_step + 1, 1)
        else:
            theslice = slice(None, None, 1)
        images = read(fname, theslice, fmt)
        if len(images) != 1:
            raise OctopusParseError("Expected only one image.  Don't know what to do with %d images." % len(images))
        return images[0]
    cell = None
    pbc = None
    adjust_positions_by_half_cell = False
    atoms = None
    xsfcoords = kwargs.pop('xsfcoordinates', None)
    if xsfcoords is not None:
        atoms = read_atoms_from_file(xsfcoords, 'xsf')
        atoms.positions *= Bohr
        atoms.cell *= Bohr
        if sum(atoms.pbc) != 3:
            raise NotImplementedError('XSF not fully periodic with Octopus')
        cell = atoms.cell
        pbc = atoms.pbc
        adjust_positions_by_half_cell = False
    xyzcoords = kwargs.pop('xyzcoordinates', None)
    if xyzcoords is not None:
        atoms = read_atoms_from_file(xyzcoords, 'xyz')
        atoms.positions *= Bohr
        adjust_positions_by_half_cell = True
    pdbcoords = kwargs.pop('pdbcoordinates', None)
    if pdbcoords is not None:
        atoms = read_atoms_from_file(pdbcoords, 'pdb')
        pbc = atoms.pbc
        adjust_positions_by_half_cell = True
        atoms.positions *= Bohr
        if sum(atoms.pbc) != 0:
            raise NotImplementedError('Periodic pdb not supported by ASE.')
    if cell is None:
        cell, kwargs = kwargs2cell(kwargs)
        if cell is not None:
            cell *= Bohr
        if cell is not None and atoms is not None:
            atoms.cell = cell
    ndims = int(kwargs.get('dimensions', 3))
    if ndims != 3:
        raise NotImplementedError('Only 3D calculations supported.')
    coords = kwargs.get('coordinates')
    if coords is not None:
        numbers, pos, tags, info = get_positions_from_block('coordinates')
        pos *= Bohr
        adjust_positions_by_half_cell = True
        atoms = Atoms(cell=cell, numbers=numbers, positions=pos, tags=tags, info=info)
    rcoords = kwargs.get('reducedcoordinates')
    if rcoords is not None:
        numbers, spos, tags, info = get_positions_from_block('reducedcoordinates')
        if cell is None:
            raise ValueError('Cannot figure out what the cell is, and thus cannot interpret reduced coordinates.')
        atoms = Atoms(cell=cell, numbers=numbers, scaled_positions=spos, tags=tags, info=info)
    if atoms is None:
        raise OctopusParseError('Apparently there are no atoms.')
    if pbc is None:
        pdims = int(kwargs.pop('periodicdimensions', 0))
        pbc = np.zeros(3, dtype=bool)
        pbc[:pdims] = True
        atoms.pbc = pbc
    if cell is not None and cell.shape == (3,) and adjust_positions_by_half_cell:
        nonpbc = atoms.pbc == 0
        atoms.positions[:, nonpbc] += np.array(cell)[None, nonpbc] / 2.0
    return (atoms, kwargs)