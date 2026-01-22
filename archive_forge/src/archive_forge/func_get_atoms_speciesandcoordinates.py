import os
import numpy as np
from ase.units import Bohr, Ha, Ry, fs, m, s
from ase.calculators.calculator import kpts2sizeandoffsets
from ase.calculators.openmx.reader import (read_electron_valency, get_file_name, get_standard_key)
from ase.calculators.openmx import parameters as param
def get_atoms_speciesandcoordinates(atoms, parameters):
    """
    The atomic coordinates and the number of spin charge are given by the
    keyword
    'Atoms.SpeciesAndCoordinates' as follows:
    <Atoms.SpeciesAndCoordinates
     1  Mn    0.00000   0.00000   0.00000   8.0  5.0  45.0 0.0 45.0 0.0  1 on
     2  O     1.70000   0.00000   0.00000   3.0  3.0  45.0 0.0 45.0 0.0  1 on
    Atoms.SpeciesAndCoordinates>
    to know more, link <http://www.openmx-square.org/openmx_man3.7/node85.html>
    """
    atoms_speciesandcoordinates = []
    xc = parameters.get('_xc')
    year = parameters.get('_year')
    data_pth = parameters.get('_data_path')
    elements = atoms.get_chemical_symbols()
    for i, element in enumerate(elements):
        atoms_speciesandcoordinates.append([str(i + 1), element])
    unit = parameters.get('atoms_speciesandcoordinates_unit', 'ang').lower()
    if unit == 'ang':
        positions = atoms.get_positions()
    elif unit == 'frac':
        positions = atoms.get_scaled_positions(wrap=False)
    elif unit == 'au':
        positions = atoms.get_positions() / Bohr
    for i, position in enumerate(positions):
        atoms_speciesandcoordinates[i].extend(position)
    if parameters.get('atoms_speciesandcoordinates') is not None:
        atoms_spncrd = parameters['atoms_speciesandcoordinates'].copy()
        for i in range(len(atoms)):
            atoms_spncrd[i][2] = atoms_speciesandcoordinates[i][2]
            atoms_spncrd[i][3] = atoms_speciesandcoordinates[i][3]
            atoms_spncrd[i][4] = atoms_speciesandcoordinates[i][4]
        return atoms_spncrd
    magmoms = atoms.get_initial_magnetic_moments()
    for i, magmom in enumerate(magmoms):
        up_down_spin = get_up_down_spin(magmom, elements[i], xc, data_pth, year)
        atoms_speciesandcoordinates[i].extend(up_down_spin)
    spin_directions = get_spin_direction(magmoms)
    for i, spin_direction in enumerate(spin_directions):
        atoms_speciesandcoordinates[i].extend(spin_direction)
    orbital_directions = get_orbital_direction()
    for i, orbital_direction in enumerate(orbital_directions):
        atoms_speciesandcoordinates[i].extend(orbital_direction)
    noncollinear_switches = get_noncollinear_switches()
    for i, noncollinear_switch in enumerate(noncollinear_switches):
        atoms_speciesandcoordinates[i].extend(noncollinear_switch)
    lda_u_switches = get_lda_u_switches()
    for i, lda_u_switch in enumerate(lda_u_switches):
        atoms_speciesandcoordinates[i].extend(lda_u_switch)
    return atoms_speciesandcoordinates