import os
import os.path
from warnings import warn
from subprocess import Popen, PIPE
import numpy as np
import ase.io
from ase.units import Rydberg
from ase.calculators.calculator import (Calculator, all_changes, Parameters,
def get_subsection(self, path):
    """Finds a subsection"""
    parts = path.upper().split('/', 1)
    candidates = [s for s in self.subsections if s.name == parts[0]]
    if len(candidates) > 1:
        raise Exception('Multiple %s sections found ' % parts[0])
    if len(candidates) == 0:
        s = InputSection(name=parts[0])
        self.subsections.append(s)
        candidates = [s]
    if len(parts) == 1:
        return candidates[0]
    return candidates[0].get_subsection(parts[1])