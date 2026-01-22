from __future__ import annotations
import logging
import os
import re
import warnings
from glob import glob
from itertools import chain
import numpy as np
import pandas as pd
from monty.io import zopen
from monty.json import MSONable, jsanitize
from monty.re import regrep
from pymatgen.core.structure import Molecule, Structure
from pymatgen.core.units import Ha_to_eV
from pymatgen.electronic_structure.bandstructure import BandStructure, BandStructureSymmLine
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.electronic_structure.dos import CompleteDos, Dos
from pymatgen.io.cp2k.inputs import Keyword
from pymatgen.io.cp2k.sets import Cp2kInput
from pymatgen.io.cp2k.utils import natural_keys, postprocessor
from pymatgen.io.xyz import XYZ
def parse_dft_params(self):
    """Parse the DFT parameters (as well as functional, HF, vdW params)."""
    pat = re.compile('\\s+DFT\\|\\s+(\\w.*)\\s\\s\\s(.*)$')
    self.read_pattern({'dft': pat}, terminate_on_match=False, postprocess=postprocessor, reverse=False)
    self.data['dft'] = dict(self.data['dft'])
    self.data['dft']['cutoffs'] = {}
    self.data['dft']['cutoffs']['density'] = self.data['dft'].pop('Cutoffs:_density', None)
    self.data['dft']['cutoffs']['gradient'] = self.data['dft'].pop('gradient', None)
    self.data['dft']['cutoffs']['tau'] = self.data['dft'].pop('tau', None)
    if self.input and self.input.check('FORCE_EVAL/DFT/XC/XC_FUNCTIONAL'):
        if (xc_funcs := list(self.input['force_eval']['dft']['xc']['xc_functional'].subsections)):
            self.data['dft']['functional'] = xc_funcs
        else:
            for v in self.input['force_eval']['dft']['xc'].subsections.values():
                if v.name.upper() == 'XC_FUNCTIONAL':
                    self.data['dft']['functional'] = v.section_parameters
    else:
        functional = re.compile('\\s+FUNCTIONAL\\|\\s+(.+):')
        self.read_pattern({'functional': functional}, terminate_on_match=False, postprocess=postprocessor, reverse=False)
        self.data['dft']['functional'] = [item for sublist in self.data.pop('functional', None) for item in sublist]
    self.data['dft']['dft_plus_u'] = self.is_hubbard
    hfx = re.compile('\\s+HFX_INFO\\|\\s+(.+):\\s+(.*)$')
    self.read_pattern({'hfx': hfx}, terminate_on_match=False, postprocess=postprocessor, reverse=False)
    self.data['dft']['hfx'] = dict(self.data.pop('hfx'))
    vdw = re.compile('\\s+vdW POTENTIAL\\|(.+)$')
    self.read_pattern({'vdw': vdw}, terminate_on_match=False, postprocess=lambda x: str(x).strip(), reverse=False)
    if self.data.get('vdw'):
        found = False
        suffix = ''
        for ll in self.data.get('vdw'):
            for _possible, _name in zip(['RVV10', 'LMKLL', 'DRSLL', 'DFT-D3', 'DFT-D2'], ['RVV10', 'LMKLL', 'DRSLL', 'D3', 'D2']):
                if _possible in ll[0]:
                    found = _name
                if 'BJ' in ll[0]:
                    suffix = '(BJ)'
        self.data['dft']['vdw'] = found + suffix if found else self.data.pop('vdw')[0][0]
    poisson_periodic = {'poisson_periodicity': re.compile('POISSON\\| Periodicity\\s+(\\w+)')}
    self.read_pattern(poisson_periodic, terminate_on_match=True)