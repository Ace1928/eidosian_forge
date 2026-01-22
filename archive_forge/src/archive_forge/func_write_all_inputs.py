import os
from os.path import join
import re
from glob import glob
import warnings
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.units import Hartree, Bohr, fs
from ase.calculators.calculator import Parameters
def write_all_inputs(atoms, properties, parameters, pp_paths=None, raise_exception=True, label='abinit', *, v8_legacy_format=True):
    species = sorted(set(atoms.numbers))
    if pp_paths is None:
        pp_paths = get_default_abinit_pp_paths()
    ppp = get_ppp_list(atoms, species, raise_exception=raise_exception, xc=parameters.xc, pps=parameters.pps, search_paths=pp_paths)
    if v8_legacy_format is None:
        warnings.warn(abinit_input_version_warning, FutureWarning)
        v8_legacy_format = True
    if v8_legacy_format:
        with open(label + '.files', 'w') as fd:
            write_files_file(fd, label, ppp)
        pseudos = None
        output_filename = label + '.txt'
    else:
        pseudos = ppp
        output_filename = label + '.abo'
    if os.path.isfile(output_filename):
        os.remove(output_filename)
    parameters.write(label + '.ase')
    with open(label + '.in', 'w') as fd:
        write_abinit_in(fd, atoms, param=parameters, species=species, pseudos=pseudos)