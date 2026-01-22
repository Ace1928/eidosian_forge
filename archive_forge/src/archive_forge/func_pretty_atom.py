import sys
import warnings
from string import ascii_lowercase, ascii_uppercase
import unicodedata
from sympy.printing.conventions import split_super_sub
from sympy.core.alphabets import greeks
from sympy.utilities.exceptions import sympy_deprecation_warning
def pretty_atom(atom_name, default=None, printer=None):
    """return pretty representation of an atom"""
    if _use_unicode:
        if printer is not None and atom_name == 'ImaginaryUnit' and (printer._settings['imaginary_unit'] == 'j'):
            return U('DOUBLE-STRUCK ITALIC SMALL J')
        else:
            return atoms_table[atom_name]
    else:
        if default is not None:
            return default
        raise KeyError('only unicode')