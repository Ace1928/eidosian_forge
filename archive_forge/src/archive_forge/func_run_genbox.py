import os
import subprocess
from glob import glob
from shutil import which
import numpy as np
from ase import units
from ase.calculators.calculator import (EnvironmentError,
from ase.io.gromos import read_gromos, write_gromos
def run_genbox(self):
    """Run gromacs program genbox, typically to solvate the system
        writing to the input structure
        as extra parameter you need to define the file containing the solvent

        for instance::

           CALC_MM_RELAX = Gromacs()
           CALC_MM_RELAX.set_own_params_runs(
                'extra_genbox_parameters', '-cs spc216.gro')
        """
    subcmd = 'genbox'
    command = ' '.join([subcmd, '-cp', self.label + '.g96', '-o', self.label + '.g96', '-p', self.label + '.top', self.params_runs.get('extra_genbox_parameters', ''), '> {}.{}.log 2>&1'.format(self.label, subcmd)])
    self._execute_gromacs(command)