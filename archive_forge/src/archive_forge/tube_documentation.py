from math import sqrt, gcd
import numpy as np
from ase.atoms import Atoms
Create an atomic structure.

    Creates a single-walled nanotube whose structure is specified using the
    standardized (n, m) notation.

    Parameters
    ----------
    n : int
        n in the (n, m) notation.
    m : int
        m in the (n, m) notation.
    length : int, optional
        Length (axial repetitions) of the nanotube.
    bond : float, optional
        Bond length between neighboring atoms.
    symbol : str, optional
        Chemical element to construct the nanotube from.
    verbose : bool, optional
        If True, will display key geometric parameters.

    Returns
    -------
    ase.atoms.Atoms
        An ASE Atoms object corresponding to the specified molecule.

    Examples
    --------
    >>> from ase.build import nanotube
    >>> atoms1 = nanotube(6, 0, length=4)
    >>> atoms2 = nanotube(3, 3, length=6, bond=1.4, symbol='Si')
    