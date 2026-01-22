from __future__ import annotations
import os
import re
import subprocess
from monty.tempfile import ScratchDir
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.core import Element, Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
@staticmethod
def keyword_line(*args):
    """Checks if the input args are proper gulp keywords and
        generates the 1st line of gulp input. Full keywords are expected.

        Args:
            args: 1st line keywords
        """
    gin = ' '.join(args)
    gin += '\n'
    return gin