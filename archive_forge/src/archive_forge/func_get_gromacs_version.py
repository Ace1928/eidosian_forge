import os
import subprocess
from glob import glob
from shutil import which
import numpy as np
from ase import units
from ase.calculators.calculator import (EnvironmentError,
from ase.io.gromos import read_gromos, write_gromos
def get_gromacs_version(executable):
    output = subprocess.check_output([executable, '--version'], encoding='utf-8')
    return parse_gromacs_version(output)