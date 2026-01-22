import os
import time
import subprocess
import re
import warnings
import numpy as np
from ase.geometry import cell_to_cellpar
from ase.calculators.calculator import (FileIOCalculator, Calculator, equal,
from ase.calculators.openmx.parameters import OpenMXParameters
from ase.calculators.openmx.default_settings import default_dictionary
from ase.calculators.openmx.reader import read_openmx, get_file_name
from ase.calculators.openmx.writer import write_openmx
def run_openmx(self):

    def isRunning(process=None):
        """ Check mpi is running"""
        return process.poll() is None
    runfile = get_file_name('.dat', self.label, absolute_directory=False)
    outfile = get_file_name('.log', self.label)
    olddir = os.getcwd()
    abs_dir = os.path.join(olddir, self.directory)
    try:
        os.chdir(abs_dir)
        if self.command is None:
            self.command = 'openmx'
        command = self.command + ' %s > %s'
        command = command % (runfile, outfile)
        self.prind(command)
        p = subprocess.Popen(command, shell=True, universal_newlines=True)
        self.print_file(file=outfile, running=isRunning, process=p)
    finally:
        os.chdir(olddir)
    self.prind('Calculation Finished')