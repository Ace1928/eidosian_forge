import os
import struct
import numpy as np
from ase.units import Ha, Bohr, Debye
from ase.io import ParseError
def get_atomic_formula(out_data=None, log_data=None, restart_data=None, scfout_data=None, dat_data=None):
    """_formula'.
    OpenMX results gives following information. Since, we should pick one
    between position/scaled_position, scaled_positions are suppressed by
    default. We use input value of position. Not the position after
    calculation. It is temporal.

       Atoms.SpeciesAndCoordinate -> symbols
       Atoms.SpeciesAndCoordinate -> positions
       Atoms.UnitVectors -> cell
       scaled_positions -> scaled_positions
        If `positions` and `scaled_positions` are both given, this key deleted
       magmoms -> magmoms, Single value for each atom or three numbers for each
                           atom for non-collinear calculations.
    """
    atomic_formula = {}
    parameters = {'symbols': list, 'positions': list, 'scaled_positions': list, 'magmoms': list, 'cell': list}
    datas = [out_data, log_data, restart_data, scfout_data, dat_data]
    atoms_unitvectors = None
    atoms_spncrd_unit = 'ang'
    atoms_unitvectors_unit = 'ang'
    for data in datas:
        if 'atoms_speciesandcoordinates_unit' in data:
            atoms_spncrd_unit = data['atoms_speciesandcoordinates_unit']
        if 'atoms_unitvectors_unit' in data:
            atoms_unitvectors_unit = data['atoms_unitvectors_unit']
        if 'atoms_speciesandcoordinates' in data:
            atoms_spncrd = data['atoms_speciesandcoordinates']
        if 'atoms_unitvectors' in data:
            atoms_unitvectors = data['atoms_unitvectors']
        if 'scf_eigenvaluesolver' in data:
            scf_eigenvaluesolver = data['scf_eigenvaluesolver']
        for openmx_keyword in data.keys():
            for standard_keyword in parameters.keys():
                if openmx_keyword == standard_keyword:
                    atomic_formula[standard_keyword] = data[openmx_keyword]
    atomic_formula['symbols'] = [i[1] for i in atoms_spncrd]
    openmx_spncrd_keyword = [[i[2], i[3], i[4]] for i in atoms_spncrd]
    positions_unit = atoms_spncrd_unit.lower()
    positions = np.array(openmx_spncrd_keyword, dtype=float)
    if positions_unit == 'ang':
        atomic_formula['positions'] = positions
    elif positions_unit == 'frac':
        scaled_positions = np.array(openmx_spncrd_keyword, dtype=float)
        atomic_formula['scaled_positions'] = scaled_positions
    elif positions_unit == 'au':
        positions = np.array(openmx_spncrd_keyword, dtype=float) * Bohr
        atomic_formula['positions'] = positions
    atomic_formula['pbc'] = scf_eigenvaluesolver.lower() != 'cluster'
    if atoms_unitvectors is not None:
        openmx_cell_keyword = atoms_unitvectors
        cell = np.array(openmx_cell_keyword, dtype=float)
        if atoms_unitvectors_unit.lower() == 'ang':
            atomic_formula['cell'] = openmx_cell_keyword
        elif atoms_unitvectors_unit.lower() == 'au':
            atomic_formula['cell'] = cell * Bohr
    if atomic_formula.get('scaled_positions') is not None and atomic_formula.get('positions') is not None:
        del atomic_formula['scaled_positions']
    return atomic_formula