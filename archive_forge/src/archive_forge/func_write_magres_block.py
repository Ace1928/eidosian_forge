import re
import numpy as np
from collections import OrderedDict
import ase.units
from ase.atoms import Atoms
from ase.spacegroup import Spacegroup
from ase.spacegroup.spacegroup import SpacegroupNotFoundError
from ase.calculators.singlepoint import SinglePointDFTCalculator
def write_magres_block(data):
    """
            Write out a <magres> block from its dictionary representation
        """
    out = []

    def nout(tag, tensor_name):
        if tag in data:
            out.append(' '.join([' ', tag, tensor_string(data[tag][tensor_name])]))

    def siout(tag, tensor_name):
        if tag in data:
            for atom_si in data[tag]:
                out.append('  %s %s %d %s' % (tag, atom_si['atom']['label'], atom_si['atom']['index'], tensor_string(atom_si[tensor_name])))
    write_units(data, out)
    nout('sus', 'S')
    siout('ms', 'sigma')
    siout('efg_local', 'V')
    siout('efg_nonlocal', 'V')
    siout('efg', 'V')

    def sisiout(tag, tensor_name):
        if tag in data:
            for isc in data[tag]:
                out.append('  %s %s %d %s %d %s' % (tag, isc['atom1']['label'], isc['atom1']['index'], isc['atom2']['label'], isc['atom2']['index'], tensor_string(isc[tensor_name])))
    sisiout('isc_fc', 'K')
    sisiout('isc_orbital_p', 'K')
    sisiout('isc_orbital_d', 'K')
    sisiout('isc_spin', 'K')
    sisiout('isc', 'K')
    return '\n'.join(out)