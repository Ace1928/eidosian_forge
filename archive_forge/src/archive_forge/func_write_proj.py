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
def write_proj(self, output_file_proj: str, output_file_def: str) -> None:
    """Writes the projections to an output file.

        Args:
            output_file_proj: output file name
            output_file_def: output file name
        """
    for oi, o in enumerate(Orbital):
        for site_nb in range(len(self._bs.structure)):
            if oi < len(self._bs.projections[Spin.up][0][0]):
                with open(f'{output_file_proj}_{site_nb}_{o}', mode='w') as file:
                    file.write(self._bs.structure.formula + '\n')
                    file.write(str(len(self._bs.kpoints)) + '\n')
                    for i, kpt in enumerate(self._bs.kpoints):
                        tmp_proj = []
                        for j in range(int(math.floor(self._bs.nb_bands * (1 - self.cb_cut)))):
                            tmp_proj.append(self._bs.projections[Spin(self.spin)][j][i][oi][site_nb])
                        if self.run_type == 'DOS' and self._bs.is_spin_polarized:
                            tmp_proj.insert(0, self._ll)
                            tmp_proj.append(self._hl)
                        a, b, c = kpt.frac_coords
                        file.write(f'{a:12.8f} {b:12.8f} {c:12.8f} {len(tmp_proj)}\n')
                        for t in tmp_proj:
                            file.write(f'{float(t):18.8f}\n')
    with open(output_file_def, mode='w') as file:
        so = ''
        if self._bs.is_spin_polarized:
            so = 'so'
        file.write(f"5, 'boltztrap.intrans',      'old',    'formatted',0\n6,'boltztrap.outputtrans',      'unknown',    'formatted',0\n20,'boltztrap.struct',         'old',    'formatted',0\n10,'boltztrap.energy{so}',         'old',    'formatted',0\n48,'boltztrap.engre',         'unknown',    'unformatted',0\n49,'boltztrap.transdos',        'unknown',    'formatted',0\n50,'boltztrap.sigxx',        'unknown',    'formatted',0\n51,'boltztrap.sigxxx',        'unknown',    'formatted',0\n21,'boltztrap.trace',           'unknown',    'formatted',0\n22,'boltztrap.condtens',           'unknown',    'formatted',0\n24,'boltztrap.halltens',           'unknown',    'formatted',0\n30,'boltztrap_BZ.cube',           'unknown',    'formatted',0\n")
        i = 1000
        for oi, o in enumerate(Orbital):
            for site_nb in range(len(self._bs.structure)):
                if oi < len(self._bs.projections[Spin.up][0][0]):
                    file.write(f"{i},'boltztrap.proj_{site_nb}_{o.name}old', 'formatted',0\n")
                    i += 1