import fractions
import functools
import re
from collections import OrderedDict
from typing import List, Tuple, Dict
import numpy as np
from scipy.spatial import ConvexHull
import ase.units as units
from ase.formula import Formula
def solvated(symbols):
    """Extract solvation energies from database.

    symbols: str
        Extract only those molecules that contain the chemical elements
        given by the symbols string (plus water and H+).

    Data from:

        Johnson JW, Oelkers EH, Helgeson HC (1992)
        Comput Geosci 18(7):899.
        doi:10.1016/0098-3004(92)90029-Q

    and:

        Pourbaix M (1966)
        Atlas of electrochemical equilibria in aqueous solutions.
        No. v. 1 in Atlas of Electrochemical Equilibria in Aqueous Solutions.
        Pergamon Press, New York.

    Returns list of (name, energy) tuples.
    """
    if isinstance(symbols, str):
        symbols = Formula(symbols).count().keys()
    if len(_solvated) == 0:
        for line in _aqueous.splitlines():
            energy, formula = line.split(',')
            name = formula + '(aq)'
            count, charge, aq = parse_formula(name)
            energy = float(energy) * 0.001 * units.kcal / units.mol
            _solvated.append((name, count, charge, aq, energy))
    references = []
    for name, count, charge, aq, energy in _solvated:
        for symbol in count:
            if symbol not in 'HO' and symbol not in symbols:
                break
        else:
            references.append((name, energy))
    return references