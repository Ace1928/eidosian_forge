import numpy as np
from ase.atoms import Atoms, symbols2numbers
from ase.data import chemical_symbols
from ase.utils import reader, writer
from .utils import verify_cell_for_export, verify_dictionary

        Return the array "name" for the given element.

        Parameters
        ----------
        name : str
            The name of the arrays. Can be any key of `atoms.arrays`
        element : str, int
            The element to be considered.
        check_same_value : bool
            Check if all values are the same in the array. Necessary for
            'occupancies' and 'debye_waller_factors' arrays.

        Returns
        -------
        array containing the values corresponding defined by "name" for the
        given element. If check_same_value, return a single element.

        