import re
import numpy as np
from ase.units import Bohr, Angstrom, Hartree, eV, Debye
def read_static_info_energy(fd, energy_unit):

    def get(name):
        for line in fd:
            if line.strip().startswith(name):
                return float(line.split('=')[-1].strip()) * energy_unit
    return dict(energy=get('Total'), free_energy=get('Free'))