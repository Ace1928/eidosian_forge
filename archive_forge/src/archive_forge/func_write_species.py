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
def write_species(self, atoms, filename):
    self.ctrlname = filename
    species_path = self.parameters.get('species_dir')
    if species_path is None:
        species_path = os.environ.get('AIMS_SPECIES_DIR')
    if species_path is None:
        raise RuntimeError('Missing species directory!  Use species_dir ' + 'parameter or set $AIMS_SPECIES_DIR environment variable.')
    control = open(filename, 'a')
    symbols = atoms.get_chemical_symbols()
    symbols2 = []
    for n, symbol in enumerate(symbols):
        if symbol not in symbols2:
            symbols2.append(symbol)
    if self.tier is not None:
        if isinstance(self.tier, int):
            self.tierlist = np.ones(len(symbols2), 'int') * self.tier
        elif isinstance(self.tier, list):
            assert len(self.tier) == len(symbols2)
            self.tierlist = self.tier
    for i, symbol in enumerate(symbols2):
        fd = os.path.join(species_path, '%02i_%s_default' % (atomic_numbers[symbol], symbol))
        reached_tiers = False
        for line in open(fd, 'r'):
            if self.tier is not None:
                if 'First tier' in line:
                    reached_tiers = True
                    self.targettier = self.tierlist[i]
                    self.foundtarget = False
                    self.do_uncomment = True
                if reached_tiers:
                    line = self.format_tiers(line)
            control.write(line)
        if self.tier is not None and (not self.foundtarget):
            raise RuntimeError('Basis tier %i not found for element %s' % (self.targettier, symbol))
        if self.parameters.get('plus_u') is not None:
            if symbol in self.parameters.plus_u.keys():
                control.write('plus_u %s \n' % self.parameters.plus_u[symbol])
    control.close()
    if self.radmul is not None:
        self.set_radial_multiplier()