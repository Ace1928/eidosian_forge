from io import StringIO
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.calculator import InputError, ReadError
from ase.calculators.calculator import CalculatorSetupError
import multiprocessing
from ase import io
import numpy as np
import json
from ase.units import Bohr, Hartree
import warnings
import os
def set_psi4(self, atoms=None):
    """
        This function sets the imported psi4 module to the settings the user
        defines
        """
    if 'PSI_SCRATCH' in os.environ:
        pass
    elif self.parameters.get('PSI_SCRATCH'):
        os.environ['PSI_SCRATCH'] = self.parameters['PSI_SCRATCH']
    if self.parameters.get('reference') is not None:
        self.psi4.set_options({'reference': self.parameters['reference']})
    if self.parameters.get('memory') is not None:
        self.psi4.set_memory(self.parameters['memory'])
    nthreads = self.parameters.get('num_threads', 1)
    if nthreads == 'max':
        nthreads = multiprocessing.cpu_count()
    self.psi4.set_num_threads(nthreads)
    if 'kpts' in self.parameters:
        raise InputError('psi4 is a non-periodic code, and thus does not require k-points. Please remove this argument.')
    if self.parameters['method'] == 'LDA':
        self.parameters['method'] = 'svwn'
    if 'nbands' in self.parameters:
        raise InputError('psi4 does not support the keyword "nbands"')
    if 'smearing' in self.parameters:
        raise InputError('Finite temperature DFT is not implemented in psi4 currently, thus a smearing argument cannot be utilized. please remove this argument')
    if 'xc' in self.parameters:
        raise InputError('psi4 does not accept the `xc` argument please use the `method` argument instead')
    if atoms is None:
        if self.atoms is None:
            return None
        else:
            atoms = self.atoms
    if self.atoms is None:
        self.atoms = atoms
    geomline = '{}\t{:.15f}\t{:.15f}\t{:.15f}'
    geom = [geomline.format(atom.symbol, *atom.position) for atom in atoms]
    geom.append('symmetry {}'.format(self.parameters['symmetry']))
    geom.append('units angstrom')
    charge = self.parameters.get('charge')
    mult = self.parameters.get('multiplicity')
    if mult is None:
        mult = 1
        if charge is not None:
            warnings.warn('A charge was provided without a spin multiplicity. A multiplicity of 1 is assumed')
    if charge is None:
        charge = 0
    geom.append('{} {}'.format(charge, mult))
    geom.append('no_reorient')
    if not os.path.isdir(self.directory):
        os.mkdir(self.directory)
    self.molecule = self.psi4.geometry('\n'.join(geom))