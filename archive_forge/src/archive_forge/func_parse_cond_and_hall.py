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
@staticmethod
def parse_cond_and_hall(path_dir, doping_levels=None):
    """Parses the conductivity and Hall tensors.

        Args:
            path_dir: Path containing .condtens / .halltens files
            doping_levels: ([float]) - doping lvls, parse outtrans to get this.

        Returns:
            mu_steps, cond, seebeck, kappa, hall, pn_doping_levels,
            mu_doping, seebeck_doping, cond_doping, kappa_doping,
            hall_doping, carrier_conc
        """
    t_steps = set()
    mu_steps = set()
    data_full = []
    data_hall = []
    data_doping_full = []
    data_doping_hall = []
    doping_levels = doping_levels or []
    with open(f'{path_dir}/boltztrap.condtens') as file:
        for line in file:
            if not line.startswith('#'):
                mu_steps.add(float(line.split()[0]))
                t_steps.add(int(float(line.split()[1])))
                data_full.append([float(c) for c in line.split()])
    with open(f'{path_dir}/boltztrap.halltens') as file:
        for line in file:
            if not line.startswith('#'):
                data_hall.append([float(c) for c in line.split()])
    if len(doping_levels) != 0:
        with open(f'{path_dir}/boltztrap.condtens_fixdoping') as file:
            for line in file:
                if not line.startswith('#') and len(line) > 2:
                    data_doping_full.append([float(c) for c in line.split()])
        with open(f'{path_dir}/boltztrap.halltens_fixdoping') as file:
            for line in file:
                if not line.startswith('#') and len(line) > 2:
                    data_doping_hall.append([float(c) for c in line.split()])
    t_steps = sorted(t_steps)
    mu_steps = sorted((Energy(m, 'Ry').to('eV') for m in mu_steps))
    cond = {t: [] for t in t_steps}
    seebeck = {t: [] for t in t_steps}
    kappa = {t: [] for t in t_steps}
    hall = {t: [] for t in t_steps}
    carrier_conc = {t: [] for t in t_steps}
    mu_doping = {'p': {t: [] for t in t_steps}, 'n': {t: [] for t in t_steps}}
    seebeck_doping = {'p': {t: [] for t in t_steps}, 'n': {t: [] for t in t_steps}}
    cond_doping = {'p': {t: [] for t in t_steps}, 'n': {t: [] for t in t_steps}}
    kappa_doping = {'p': {t: [] for t in t_steps}, 'n': {t: [] for t in t_steps}}
    hall_doping = {'p': {t: [] for t in t_steps}, 'n': {t: [] for t in t_steps}}
    pn_doping_levels = {'p': [], 'n': []}
    for d in doping_levels:
        if d > 0:
            pn_doping_levels['p'].append(d)
        else:
            pn_doping_levels['n'].append(-d)
    for d in data_full:
        temp, doping = (d[1], d[2])
        carrier_conc[temp].append(doping)
        cond[temp].append(np.reshape(d[3:12], (3, 3)).tolist())
        seebeck[temp].append(np.reshape(d[12:21], (3, 3)).tolist())
        kappa[temp].append(np.reshape(d[21:30], (3, 3)).tolist())
    for d in data_hall:
        temp, doping = (d[1], d[2])
        hall_tens = [np.reshape(d[3:12], (3, 3)).tolist(), np.reshape(d[12:21], (3, 3)).tolist(), np.reshape(d[21:30], (3, 3)).tolist()]
        hall[temp].append(hall_tens)
    for d in data_doping_full:
        temp, doping, mu = (d[0], d[1], d[-1])
        pn = 'p' if doping > 0 else 'n'
        mu_doping[pn][temp].append(Energy(mu, 'Ry').to('eV'))
        cond_doping[pn][temp].append(np.reshape(d[2:11], (3, 3)).tolist())
        seebeck_doping[pn][temp].append(np.reshape(d[11:20], (3, 3)).tolist())
        kappa_doping[pn][temp].append(np.reshape(d[20:29], (3, 3)).tolist())
    for d in data_doping_hall:
        temp, doping, mu = (d[0], d[1], d[-1])
        pn = 'p' if doping > 0 else 'n'
        hall_tens = [np.reshape(d[2:11], (3, 3)).tolist(), np.reshape(d[11:20], (3, 3)).tolist(), np.reshape(d[20:29], (3, 3)).tolist()]
        hall_doping[pn][temp].append(hall_tens)
    return (mu_steps, cond, seebeck, kappa, hall, pn_doping_levels, mu_doping, seebeck_doping, cond_doping, kappa_doping, hall_doping, carrier_conc)