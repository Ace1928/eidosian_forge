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
def write_intrans(self, output_file) -> None:
    """Writes the intrans to an output file.

        Args:
            output_file: Filename
        """
    setgap = 1 if self.scissor > 0.0001 else 0
    if self.run_type in ('BOLTZ', 'DOS'):
        with open(output_file, mode='w') as fout:
            fout.write('GENE          # use generic interface\n')
            fout.write(f'1 0 {setgap} {Energy(self.scissor, 'eV').to('Ry')}         # iskip (not presently used) idebug setgap shiftgap \n')
            fout.write(f'0.0 {Energy(self.energy_grid, 'eV').to('Ry')} {Energy(self.energy_span_around_fermi, 'eV').to('Ry')} {self._nelec}.1f     # Fermilevel (Ry),energygrid,energy span around Fermilevel, number of electrons\n')
            fout.write('CALC                    # CALC (calculate expansion coeff), NOCALC read from file\n')
            fout.write(f'{self.lpfac}                        # lpfac, number of latt-points per k-point\n')
            fout.write(f'{self.run_type}                     # run mode (only BOLTZ is supported)\n')
            fout.write('.15                       # (efcut) energy range of chemical potential\n')
            fout.write(f'{self.tmax} {self.tgrid}                  # Tmax, temperature grid\n')
            fout.write('-1.  # energyrange of bands given DOS output sig_xxx and dos_xxx (xxx is band number)\n')
            fout.write(self.dos_type + '\n')
            fout.write(f'{self.tauref} {self.tauexp} {self.tauen} 0 0 0\n')
            fout.write(f'{2 * len(self.doping)}\n')
            for d in self.doping:
                fout.write(str(d) + '\n')
            for d in self.doping:
                fout.write(str(-d) + '\n')
    elif self.run_type == 'FERMI':
        with open(output_file, mode='w') as fout:
            fout.write('GENE          # use generic interface\n')
            fout.write('1 0 0 0.0         # iskip (not presently used) idebug setgap shiftgap \n')
            fout.write(f'0.0 {Energy(self.energy_grid, 'eV').to('Ry')} 0.1 {self._nelec:6.1f}     # Fermilevel (Ry),energygrid,energy span around Fermilevel, number of electrons\n')
            fout.write('CALC                    # CALC (calculate expansion coeff), NOCALC read from file\n')
            fout.write(f'{self.lpfac}                        # lpfac, number of latt-points per k-point\n')
            fout.write('FERMI                     # run mode (only BOLTZ is supported)\n')
            fout.write(f'1                        # actual band selected: {self.band_nb + 1} spin: {self.spin}')
    elif self.run_type == 'BANDS':
        if self.kpt_line is None:
            kpath = HighSymmKpath(self._bs.structure)
            self.kpt_line = [Kpoint(k, self._bs.structure.lattice) for k in kpath.get_kpoints(coords_are_cartesian=False)[0]]
            self.kpt_line = [kp.frac_coords for kp in self.kpt_line]
        elif isinstance(self.kpt_line[0], Kpoint):
            self.kpt_line = [kp.frac_coords for kp in self.kpt_line]
        with open(output_file, mode='w') as fout:
            fout.write('GENE          # use generic interface\n')
            fout.write(f'1 0 {setgap} {Energy(self.scissor, 'eV').to('Ry')}         # iskip (not presently used) idebug setgap shiftgap \n')
            fout.write(f'0.0 {Energy(self.energy_grid, 'eV').to('Ry')} {Energy(self.energy_span_around_fermi, 'eV').to('Ry')} {self._nelec:.1f}     # Fermilevel (Ry),energygrid,energy span around Fermilevel, number of electrons\n')
            fout.write('CALC                    # CALC (calculate expansion coeff), NOCALC read from file\n')
            fout.write(f'{self.lpfac}                        # lpfac, number of latt-points per k-point\n')
            fout.write('BANDS                     # run mode (only BOLTZ is supported)\n')
            fout.write(f'P {len(self.kpt_line)}\n')
            for kp in self.kpt_line:
                fout.writelines([f'{k} ' for k in kp])
                fout.write('\n')