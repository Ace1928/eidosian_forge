from __future__ import annotations
from itertools import chain, combinations
from pymatgen.core import Element
from pymatgen.core.composition import Composition
Fill up the atomic orbitals with available electrons.

        Returns:
            HOMO, LUMO, and whether it's a metal.
        