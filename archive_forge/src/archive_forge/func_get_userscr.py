import os
import re
from subprocess import call, TimeoutExpired
from copy import deepcopy
import numpy as np
from ase import Atoms
from ase.utils import workdir
from ase.units import Hartree, Bohr, Debye
from ase.calculators.singlepoint import SinglePointCalculator
def get_userscr(prefix, command):
    prefix_test = prefix + '_test'
    command = command.replace('PREFIX', prefix_test)
    with workdir(prefix_test, mkdir=True):
        try:
            call(command, shell=True, timeout=2)
        except TimeoutExpired:
            pass
        try:
            with open(prefix_test + '.log') as fd:
                for line in fd:
                    if line.startswith('GAMESS supplementary output files'):
                        return ' '.join(line.split(' ')[8:]).strip()
        except FileNotFoundError:
            return None
    return None