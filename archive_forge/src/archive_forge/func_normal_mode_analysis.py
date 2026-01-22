import os
import re
import warnings
from subprocess import Popen, PIPE
from math import log10, floor
import numpy as np
from ase import Atoms
from ase.units import Ha, Bohr
from ase.io import read, write
from ase.calculators.calculator import FileIOCalculator
from ase.calculators.calculator import PropertyNotImplementedError, ReadError
def normal_mode_analysis(self, atoms=None):
    """execute normal mode analysis with modules aoforce or NumForce"""
    from ase.constraints import FixAtoms
    if atoms is None:
        atoms = self.atoms
    self.set_atoms(atoms)
    self.initialize()
    if self.update_energy:
        self.get_potential_energy(atoms)
    if self.update_hessian:
        fixatoms = []
        for constr in atoms.constraints:
            if isinstance(constr, FixAtoms):
                ckwargs = constr.todict()['kwargs']
                if 'indices' in ckwargs.keys():
                    fixatoms.extend(ckwargs['indices'])
        if self.parameters['numerical hessian'] is None:
            if len(fixatoms) > 0:
                define_str = '\n\ny\n'
                for index in fixatoms:
                    define_str += 'm ' + str(index + 1) + ' 999.99999999\n'
                define_str += '*\n*\nn\nq\n'
                execute('define', input_str=define_str)
                dg = read_data_group('atoms')
                regex = '(mass\\s*=\\s*)999.99999999'
                dg = re.sub(regex, '\\g<1>9999999999.9', dg)
                dg += '\n'
                delete_data_group('atoms')
                add_data_group(dg, raw=True)
            execute('aoforce')
        else:
            optstr = ''
            pdict = self.parameters['numerical hessian']
            if self.parameters['use resolution of identity']:
                optstr += ' -ri'
            if len(fixatoms) > 0:
                optstr += ' -frznuclei -central -c'
            if 'central' in pdict.keys():
                optstr += ' -central'
            if 'delta' in pdict.keys():
                optstr += ' -d ' + str(pdict['delta'] / Bohr)
            execute('NumForce' + optstr)
        self.update_hessian = False