import os
import numpy as np
from ase.units import Bohr, Ha, Ry, fs, m, s
from ase.calculators.calculator import kpts2sizeandoffsets
from ase.calculators.openmx.reader import (read_electron_valency, get_file_name, get_standard_key)
from ase.calculators.openmx import parameters as param
def get_cutoff_radius_and_orbital(element=None, orbital=None):
    """
    For a given element, retruns the string specifying cutoff radius and
    orbital using default_settings.py. For example,
       'Si'   ->   'Si.7.0-s2p2d1'
    If one wannts to change the atomic radius for a special purpose, one should
    change the default_settings.py directly.
    """
    from ase.calculators.openmx import default_settings
    orbital = element
    orbital_letters = ['s', 'p', 'd', 'f', 'g', 'h']
    default_dictionary = default_settings.default_dictionary
    orbital_numbers = default_dictionary[element]['orbitals used']
    cutoff_radius = default_dictionary[element]['cutoff radius']
    orbital += '%.1f' % float(cutoff_radius) + '-'
    for i, orbital_number in enumerate(orbital_numbers):
        orbital += orbital_letters[i] + str(orbital_number)
    return orbital