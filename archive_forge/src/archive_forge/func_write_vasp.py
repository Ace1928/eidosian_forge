import re
import numpy as np
from ase import Atoms
from ase.utils import reader, writer
from ase.io.utils import ImageIterator
from ase.io import ParseError
from .vasp_parsers import vasp_outcar_parsers as vop
from pathlib import Path
@writer
def write_vasp(filename, atoms, label=None, direct=False, sort=None, symbol_count=None, long_format=True, vasp5=True, ignore_constraints=False, wrap=False):
    """Method to write VASP position (POSCAR/CONTCAR) files.

    Writes label, scalefactor, unitcell, # of various kinds of atoms,
    positions in cartesian or scaled coordinates (Direct), and constraints
    to file. Cartesian coordinates is default and default label is the
    atomic species, e.g. 'C N H Cu'.
    """
    from ase.constraints import FixAtoms, FixScaled, FixedPlane, FixedLine
    fd = filename
    if isinstance(atoms, (list, tuple)):
        if len(atoms) > 1:
            raise RuntimeError("Don't know how to save more than " + 'one image to VASP input')
        else:
            atoms = atoms[0]
    if np.any(atoms.cell.cellpar() == 0.0):
        raise RuntimeError('Lattice vectors must be finite and not coincident. At least one lattice length or angle is zero.')
    if direct:
        coord = atoms.get_scaled_positions(wrap=wrap)
    else:
        coord = atoms.get_positions(wrap=wrap)
    constraints = atoms.constraints and (not ignore_constraints)
    if constraints:
        sflags = np.zeros((len(atoms), 3), dtype=bool)
        for constr in atoms.constraints:
            if isinstance(constr, FixScaled):
                sflags[constr.a] = constr.mask
            elif isinstance(constr, FixAtoms):
                sflags[constr.index] = [True, True, True]
            elif isinstance(constr, FixedPlane):
                mask = np.all(np.abs(np.cross(constr.dir, atoms.cell)) < 1e-05, axis=1)
                if sum(mask) != 1:
                    raise RuntimeError('VASP requires that the direction of FixedPlane constraints is parallel with one of the cell axis')
                sflags[constr.a] = mask
            elif isinstance(constr, FixedLine):
                mask = np.all(np.abs(np.cross(constr.dir, atoms.cell)) < 1e-05, axis=1)
                if sum(mask) != 1:
                    raise RuntimeError('VASP requires that the direction of FixedLine constraints is parallel with one of the cell axis')
                sflags[constr.a] = ~mask
    if sort:
        ind = np.argsort(atoms.get_chemical_symbols())
        symbols = np.array(atoms.get_chemical_symbols())[ind]
        coord = coord[ind]
        if constraints:
            sflags = sflags[ind]
    else:
        symbols = atoms.get_chemical_symbols()
    if symbol_count:
        sc = symbol_count
    else:
        sc = _symbol_count_from_symbols(symbols)
    if label is None:
        label = ''
        for sym, c in sc:
            label += '%2s ' % sym
    fd.write(label + '\n')
    fd.write('%19.16f\n' % 1.0)
    if long_format:
        latt_form = ' %21.16f'
    else:
        latt_form = ' %11.6f'
    for vec in atoms.get_cell():
        fd.write(' ')
        for el in vec:
            fd.write(latt_form % el)
        fd.write('\n')
    _write_symbol_count(fd, sc, vasp5=vasp5)
    if constraints:
        fd.write('Selective dynamics\n')
    if direct:
        fd.write('Direct\n')
    else:
        fd.write('Cartesian\n')
    if long_format:
        cform = ' %19.16f'
    else:
        cform = ' %9.6f'
    for iatom, atom in enumerate(coord):
        for dcoord in atom:
            fd.write(cform % dcoord)
        if constraints:
            for flag in sflags[iatom]:
                if flag:
                    s = 'F'
                else:
                    s = 'T'
                fd.write('%4s' % s)
        fd.write('\n')