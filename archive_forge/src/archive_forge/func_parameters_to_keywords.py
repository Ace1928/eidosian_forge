import os
import numpy as np
from ase.units import Bohr, Ha, Ry, fs, m, s
from ase.calculators.calculator import kpts2sizeandoffsets
from ase.calculators.openmx.reader import (read_electron_valency, get_file_name, get_standard_key)
from ase.calculators.openmx import parameters as param
def parameters_to_keywords(label=None, atoms=None, parameters=None, properties=None, system_changes=None):
    """
    Before writing `label.dat` file, set up the ASE variables to OpenMX
    keywords. First, It initializes with given openmx keywords and reconstruct
    dictionary using standard parameters. If standard parameters and openmx
    keywords are contradict to each other, ignores openmx keyword.
     It includes,

    For aesthetical purpose, sequnece of writing input file is specified.
    """
    from ase.calculators.openmx.parameters import matrix_keys
    from ase.calculators.openmx.parameters import unit_dat_keywords
    from collections import OrderedDict
    keywords = OrderedDict()
    sequence = ['system_currentdirectory', 'system_name', 'data_path', 'level_of_fileout', 'species_number', 'definition_of_atomic_species', 'atoms_number', 'atoms_speciesandcoordinates_unit', 'atoms_speciesandcoordinates', 'atoms_unitvectors_unit', 'atoms_unitvectors', 'band_dispersion', 'band_nkpath', 'band_kpath']
    directory, prefix = os.path.split(label)
    curdir = os.path.join(os.getcwd(), prefix)
    counterparts = {'system_currentdirectory': curdir, 'system_name': prefix, 'data_path': os.environ.get('OPENMX_DFT_DATA_PATH'), 'species_number': len(get_species(atoms.get_chemical_symbols())), 'atoms_number': len(atoms), 'scf_restart': 'restart', 'scf_maxiter': 'maxiter', 'scf_xctype': 'xc', 'scf_energycutoff': 'energy_cutoff', 'scf_criterion': 'convergence', 'scf_external_fields': 'external', 'scf_mixing_type': 'mixer', 'scf_electronic_temperature': 'smearing', 'scf_system_charge': 'charge', 'scf_eigenvaluesolver': 'eigensolver'}
    standard_units = {'eV': 1, 'Ha': Ha, 'Ry': Ry, 'Bohr': Bohr, 'fs': fs, 'K': 1, 'GV / m': 1000000000.0 / 1.6e-19 / m, 'Ha/Bohr': Ha / Bohr, 'm/s': m / s, '_amu': 1, 'Tesla': 1}
    unit_dict = {get_standard_key(k): v for k, v in unit_dat_keywords.items()}
    for key in sequence:
        keywords[key] = None
    for key in parameters:
        if 'scf' in key:
            keywords[key] = None
    for key in parameters:
        if 'md' in key:
            keywords[key] = None
    for key in parameters.keys():
        keywords[key] = parameters[key]

    def parameter_overwrites(openmx_keyword):
        """
        In a situation conflicting ASE standard parameters and OpenMX keywords,
        ASE parameters overrides to OpenMX keywords. While doing so, units are
        converted to OpenMX unit.
        However, if both parameters and keyword are not given, we fill up that
        part in suitable manner
          openmx_keyword : key |  Name of key used in OpenMX
          keyword : value | value corresponds to openmx_keyword
          ase_parameter : key | Name of parameter used in ASE
          parameter : value | value corresponds to ase_parameter
        """
        ase_parameter = counterparts[openmx_keyword]
        keyword = parameters.get(openmx_keyword)
        parameter = parameters.get(ase_parameter)
        if parameter is not None:
            unit = standard_units.get(unit_dict.get(openmx_keyword))
            if unit is not None:
                return parameter / unit
            return parameter
        elif keyword is not None:
            return keyword
        elif 'scf' in openmx_keyword:
            return None
        else:
            return counterparts[openmx_keyword]
    for openmx_keyword in counterparts.keys():
        keywords[openmx_keyword] = parameter_overwrites(openmx_keyword)
    if 'energies' in properties:
        keywords['energy_decomposition'] = True
    if 'stress' in properties:
        keywords['scf_stress_tensor'] = True
    keywords['scf_xctype'] = get_xc(keywords['scf_xctype'])
    keywords['scf_kgrid'] = get_scf_kgrid(atoms, parameters)
    keywords['scf_spinpolarization'] = get_spinpol(atoms, parameters)
    if parameters.get('band_kpath') is not None:
        keywords['band_dispersion'] = True
    keywords['band_kpath'] = parameters.get('band_kpath')
    if parameters.get('band_nkpath') is not None:
        keywords['band_nkpath'] = len(keywords['band_kpath'])
    if parameters.get('wannier_func_calc') is not None:
        keywords['species_number'] *= 2
    parameters['_xc'] = keywords['scf_xctype']
    parameters['_data_path'] = keywords['data_path']
    parameters['_year'] = get_dft_data_year(parameters)
    for key in matrix_keys:
        get_matrix_key = globals()['get_' + get_standard_key(key)]
        keywords[get_standard_key(key)] = get_matrix_key(atoms, parameters)
    return OrderedDict([(k, v) for k, v in keywords.items() if not (v is None or (isinstance(v, list) and v == []))])