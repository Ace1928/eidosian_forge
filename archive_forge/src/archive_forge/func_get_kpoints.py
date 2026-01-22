import os
import time
import subprocess
import re
import warnings
import numpy as np
from ase.geometry import cell_to_cellpar
from ase.calculators.calculator import (FileIOCalculator, Calculator, equal,
from ase.calculators.openmx.parameters import OpenMXParameters
from ase.calculators.openmx.default_settings import default_dictionary
from ase.calculators.openmx.reader import read_openmx, get_file_name
from ase.calculators.openmx.writer import write_openmx
def get_kpoints(self, kpts=None, symbols=None, band_kpath=None, eps=1e-05):
    """Convert band_kpath <-> kpts"""
    if kpts is None:
        kpts = []
        band_kpath = np.array(band_kpath)
        band_nkpath = len(band_kpath)
        for i, kpath in enumerate(band_kpath):
            end = False
            nband = int(kpath[0])
            if band_nkpath == i:
                end = True
                nband += 1
            ini = np.array(kpath[1:4], dtype=float)
            fin = np.array(kpath[4:7], dtype=float)
            x = np.linspace(ini[0], fin[0], nband, endpoint=end)
            y = np.linspace(ini[1], fin[1], nband, endpoint=end)
            z = np.linspace(ini[2], fin[2], nband, endpoint=end)
            kpts.extend(np.array([x, y, z]).T)
        return np.array(kpts, dtype=float)
    elif band_kpath is None:
        band_kpath = []
        points = np.asarray(kpts)
        diffs = points[1:] - points[:-1]
        kinks = abs(diffs[1:] - diffs[:-1]).sum(1) > eps
        N = len(points)
        indices = [0]
        indices.extend(np.arange(1, N - 1)[kinks])
        indices.append(N - 1)
        for start, end, s_sym, e_sym in zip(indices[1:], indices[:-1], symbols[1:], symbols[:-1]):
            band_kpath.append({'start_point': start, 'end_point': end, 'kpts': 20, 'path_symbols': (s_sym, e_sym)})
        return band_kpath