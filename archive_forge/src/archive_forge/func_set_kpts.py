import difflib
import numpy as np
import os
import re
import glob
import shutil
import sys
import json
import time
import tempfile
import warnings
import subprocess
from copy import deepcopy
from collections import namedtuple
from itertools import product
from typing import List, Set
import ase
import ase.units as units
from ase.calculators.general import Calculator
from ase.calculators.calculator import compare_atoms
from ase.calculators.calculator import PropertyNotImplementedError
from ase.calculators.calculator import kpts2sizeandoffsets
from ase.dft.kpoints import BandPath
from ase.parallel import paropen
from ase.io.castep import read_param
from ase.io.castep import read_bands
from ase.constraints import FixCartesian
def set_kpts(self, kpts):
    """Set k-point mesh/path using a str, tuple or ASE features

        Args:
            kpts (None, tuple, str, dict):

        This method will set the CASTEP parameters kpoints_mp_grid,
        kpoints_mp_offset and kpoints_mp_spacing as appropriate. Unused
        parameters will be set to None (i.e. not included in input files.)

        If kpts=None, all these parameters are set as unused.

        The simplest useful case is to give a 3-tuple of integers specifying
        a Monkhorst-Pack grid. This may also be formatted as a string separated
        by spaces; this is the format used internally before writing to the
        input files.

        A more powerful set of features is available when using a python
        dictionary with the following allowed keys:

        - 'size' (3-tuple of int) mesh of mesh dimensions
        - 'density' (float) for BZ sampling density in points per recip. Ang
          ( kpoint_mp_spacing = 1 / (2pi * density) ). An explicit MP mesh will
          be set to allow for rounding/centering.
        - 'spacing' (float) for BZ sampling density for maximum space between
          sample points in reciprocal space. This is numerically equivalent to
          the inbuilt ``calc.cell.kpoint_mp_spacing``, but will be converted to
          'density' to allow for rounding/centering.
        - 'even' (bool) to round each direction up to the nearest even number;
          set False for odd numbers, leave as None for no odd/even rounding.
        - 'gamma' (bool) to offset the Monkhorst-Pack grid to include
          (0, 0, 0); set False to offset each direction avoiding 0.
        """

    def clear_mp_keywords():
        mp_keywords = product({'kpoint', 'kpoints'}, {'mp_grid', 'mp_offset', 'mp_spacing', 'list'})
        for kp_tag in mp_keywords:
            setattr(self.cell, '_'.join(kp_tag), None)
    if kpts is None:
        clear_mp_keywords()
        pass
    elif isinstance(kpts, (tuple, list)) and isinstance(kpts[0], (tuple, list)):
        if not all(map(lambda row: len(row) == 4, kpts)):
            raise ValueError('In explicit kpt list each row should have 4 elements')
        clear_mp_keywords()
        self.cell.kpoint_list = [' '.join(map(str, row)) for row in kpts]
    elif isinstance(kpts, (tuple, list)) and isinstance(kpts[0], str):
        if not all(map(lambda row: len(row.split()) == 4, kpts)):
            raise ValueError('In explicit kpt list each row should have 4 elements')
        clear_mp_keywords()
        self.cell.kpoint_list = kpts
    elif isinstance(kpts, (tuple, list)) and isinstance(kpts[0], int):
        if len(kpts) != 3:
            raise ValueError('Monkhorst-pack grid should have 3 values')
        clear_mp_keywords()
        self.cell.kpoint_mp_grid = '%d %d %d' % tuple(kpts)
    elif isinstance(kpts, str):
        self.set_kpts([int(x) for x in kpts.split()])
    elif isinstance(kpts, dict):
        kpts = kpts.copy()
        if kpts.get('spacing') is not None and kpts.get('density') is not None:
            raise ValueError('Cannot set kpts spacing and density simultaneously.')
        else:
            if kpts.get('spacing') is not None:
                kpts = kpts.copy()
                spacing = kpts.pop('spacing')
                kpts['density'] = 1 / (np.pi * spacing)
            clear_mp_keywords()
            size, offsets = kpts2sizeandoffsets(atoms=self.atoms, **kpts)
            self.cell.kpoint_mp_grid = '%d %d %d' % tuple(size)
            self.cell.kpoint_mp_offset = '%f %f %f' % tuple(offsets)
    elif hasattr(kpts, '__iter__'):
        self.set_kpts(list(kpts))
    else:
        raise TypeError('Cannot interpret kpts of this type')