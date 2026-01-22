import os
import re
import numpy as np
from ase import Atoms
from ase.io import read
from ase.io.dmol import write_dmol_car, write_dmol_incoor
from ase.units import Hartree, Bohr
from ase.calculators.calculator import FileIOCalculator, Parameters, ReadError
def finished_successfully(self):
    """ Reads outmol file and checks if job completed or failed.

        Returns
        -------
        finished (bool): True if job completed, False if something went wrong
        message (str): If job failed message contains parsed errors, else empty

        """
    finished = False
    message = ''
    for line in self._outmol_lines():
        if line.rfind('Message: DMol3 job finished successfully') > -1:
            finished = True
        if line.startswith('Error'):
            message += line
    return (finished, message)