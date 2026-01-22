import os
import os.path as op
import subprocess
import shutil
import numpy as np
from ase.units import Bohr, Hartree
import ase.data
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.calculators.calculator import Parameters, all_changes
from ase.calculators.calculator import equal
import ase.io
from .demon_io import parse_xray
def link_file(self, fromdir, todir, filename):
    if op.exists(todir + '/' + filename):
        os.remove(todir + '/' + filename)
    if op.exists(fromdir + '/' + filename):
        os.symlink(fromdir + '/' + filename, todir + '/' + filename)
    else:
        raise RuntimeError("{0} doesn't exist".format(fromdir + '/' + filename))