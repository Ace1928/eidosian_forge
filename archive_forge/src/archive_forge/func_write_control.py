import os
import warnings
import time
from typing import Optional
import re
import numpy as np
from ase.units import Hartree
from ase.io.aims import write_aims, read_aims
from ase.data import atomic_numbers
from ase.calculators.calculator import FileIOCalculator, Parameters, kpts2mp, \
def write_control(self, atoms, filename, debug=False):
    lim = '#' + '=' * 79
    output = open(filename, 'w')
    output.write(lim + '\n')
    for line in ['FHI-aims file: ' + filename, 'Created using the Atomic Simulation Environment (ASE)', time.asctime()]:
        output.write('# ' + line + '\n')
    if debug:
        output.write('# \n# List of parameters used to initialize the calculator:')
        for p, v in self.parameters.items():
            s = '#     {} : {}\n'.format(p, v)
            output.write(s)
    output.write(lim + '\n')
    assert not ('kpts' in self.parameters and 'k_grid' in self.parameters)
    assert not ('smearing' in self.parameters and 'occupation_type' in self.parameters)
    for key, value in self.parameters.items():
        if key == 'kpts':
            mp = kpts2mp(atoms, self.parameters.kpts)
            output.write('%-35s%d %d %d\n' % (('k_grid',) + tuple(mp)))
            dk = 0.5 - 0.5 / np.array(mp)
            output.write('%-35s%f %f %f\n' % (('k_offset',) + tuple(dk)))
        elif key == 'species_dir' or key == 'run_command':
            continue
        elif key == 'plus_u':
            continue
        elif key == 'smearing':
            name = self.parameters.smearing[0].lower()
            if name == 'fermi-dirac':
                name = 'fermi'
            width = self.parameters.smearing[1]
            output.write('%-35s%s %f' % ('occupation_type', name, width))
            if name == 'methfessel-paxton':
                order = self.parameters.smearing[2]
                output.write(' %d' % order)
            output.write('\n' % order)
        elif key == 'output':
            for output_type in value:
                output.write('%-35s%s\n' % (key, output_type))
        elif key == 'vdw_correction_hirshfeld' and value:
            output.write('%-35s\n' % key)
        elif key in bool_keys:
            output.write('%-35s.%s.\n' % (key, repr(bool(value)).lower()))
        elif isinstance(value, (tuple, list)):
            output.write('%-35s%s\n' % (key, ' '.join((str(x) for x in value))))
        elif isinstance(value, str):
            output.write('%-35s%s\n' % (key, value))
        else:
            output.write('%-35s%r\n' % (key, value))
    if self.cubes:
        self.cubes.write(output)
    output.write(lim + '\n\n')
    output.close()