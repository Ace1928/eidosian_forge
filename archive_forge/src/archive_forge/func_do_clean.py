import os
import subprocess
from glob import glob
from shutil import which
import numpy as np
from ase import units
from ase.calculators.calculator import (EnvironmentError,
from ase.io.gromos import read_gromos, write_gromos
def do_clean(name='#*'):
    """ remove files matching wildcards """
    myfiles = glob(name)
    for myfile in myfiles:
        try:
            os.remove(myfile)
        except OSError:
            pass