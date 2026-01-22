from collections.abc import Mapping, Sequence
from subprocess import check_call, DEVNULL
from os import unlink
from pathlib import Path
import numpy as np
from ase.io.utils import PlottingVariables
from ase.constraints import FixAtoms
from ase import Atoms
@staticmethod
def wrapped_triples_section(triple_list, triple_format='<{:f}, {:f}, {:f}>'.format, triples_per_line=4):
    triples = [triple_format(*x) for x in triple_list]
    n = len(triples)
    s = ''
    tpl = triples_per_line
    c = 0
    while c < n - tpl:
        c += tpl
        s += '\n     '
        s += ', '.join(triples[c - tpl:c])
    s += '\n    '
    s += ', '.join(triples[c:])
    return s