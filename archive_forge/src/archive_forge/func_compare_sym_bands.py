from __future__ import annotations
import logging
import math
import os
import subprocess
import tempfile
import time
from shutil import which
from typing import TYPE_CHECKING, Literal
import numpy as np
from monty.dev import requires
from monty.json import MSONable, jsanitize
from monty.os import cd
from scipy import constants
from scipy.optimize import fsolve
from scipy.spatial import distance
from pymatgen.core.lattice import Lattice
from pymatgen.core.units import Energy, Length
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine, Kpoint
from pymatgen.electronic_structure.core import Orbital
from pymatgen.electronic_structure.dos import CompleteDos, Dos, Spin
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
def compare_sym_bands(bands_obj, bands_ref_obj, nb=None):
    """Compute the mean of correlation between bzt and vasp bandstructure on
    sym line, for all bands and locally (for each branches) the difference
    squared (%) if nb is specified.
    """
    if bands_ref_obj.is_spin_polarized:
        nbands = min(bands_obj.nb_bands, 2 * bands_ref_obj.nb_bands)
    else:
        nbands = min(len(bands_obj.bands[Spin.up]), len(bands_ref_obj.bands[Spin.up]))
    arr_bands = np.array(bands_obj.bands[Spin.up][:nbands])
    if bands_ref_obj.is_spin_polarized:
        arr_bands_ref_up = np.array(bands_ref_obj.bands[Spin.up])
        arr_bands_ref_dw = np.array(bands_ref_obj.bands[Spin.down])
        arr_bands_ref = np.vstack((arr_bands_ref_up, arr_bands_ref_dw))
        arr_bands_ref = np.sort(arr_bands_ref, axis=0)[:nbands]
    else:
        arr_bands_ref = np.array(bands_ref_obj.bands[Spin.up][:nbands])
    corr = np.array([distance.correlation(arr_bands[idx], arr_bands_ref[idx]) for idx in range(nbands)])
    if isinstance(nb, int):
        nb = [nb]
    bcheck = {}
    if max(nb) < nbands:
        branches = [[s['start_index'], s['end_index'], s['name']] for s in bands_ref_obj.branches]
        if not bands_obj.is_metal() and (not bands_ref_obj.is_metal()):
            zero_ref = bands_ref_obj.get_vbm()['energy']
            zero = bands_obj.get_vbm()['energy']
            if not zero:
                vbm = bands_ref_obj.get_vbm()['band_index'][Spin.up][-1]
                zero = max(arr_bands[vbm])
        else:
            zero_ref = 0
            zero = 0
            print(zero, zero_ref)
        for nbi in nb:
            bcheck[nbi] = {}
            bcheck[nbi]['Dist'] = np.mean(abs(arr_bands[nbi] - zero - arr_bands_ref[nbi] + zero_ref))
            bcheck[nbi]['Corr'] = corr[nbi]
            for start, end, name in branches:
                bcheck[nbi][name] = np.mean(abs(arr_bands[nbi][start:end + 1] - zero - arr_bands_ref[nbi][start:end + 1] + zero_ref))
    else:
        bcheck = 'No nb given'
    return bcheck