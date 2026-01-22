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
def read_xray(self):
    """Read deMon.xry if present."""
    filename = self.label + '/deMon.out'
    core_IP = None
    if op.isfile(filename):
        with open(filename, 'r') as fd:
            lines = fd.readlines()
        for i in range(len(lines)):
            if lines[i].rfind('IONIZATION POTENTIAL') > -1:
                core_IP = float(lines[i].split()[3])
    try:
        mode, ntrans, E_trans, osc_strength, trans_dip = parse_xray(self.label + '/deMon.xry')
    except ReadError:
        pass
    else:
        xray_results = {'xray_mode': mode, 'ntrans': ntrans, 'E_trans': E_trans, 'osc_strength': osc_strength, 'trans_dip': trans_dip, 'core_IP': core_IP}
        self.results['xray'] = xray_results