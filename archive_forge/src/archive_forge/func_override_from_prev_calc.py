from __future__ import annotations
import abc
import itertools
import os
import re
import shutil
import warnings
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from glob import glob
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Union, cast
from zipfile import ZipFile
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.json import MSONable
from monty.serialization import loadfn
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Element, PeriodicSite, SiteCollection, Species, Structure
from pymatgen.io.core import InputGenerator
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar, VaspInput
from pymatgen.io.vasp.outputs import Outcar, Vasprun
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.util.due import Doi, due
def override_from_prev_calc(self, prev_calc_dir='.'):
    """
        Update the input set to include settings from a previous calculation.

        Args:
            prev_calc_dir (str): The path to the previous calculation directory.

        Returns:
            The input set with the settings (structure, k-points, incar, etc)
            updated using the previous VASP run.
        """
    self._set_previous(prev_calc_dir)
    if self.standardize:
        warnings.warn('Use of standardize=True with from_prev_run is not recommended as there is no guarantee the copied files will be appropriate for the standardized structure.')
    files_to_transfer = {}
    if getattr(self, 'copy_chgcar', False):
        chgcars = sorted(glob(str(Path(prev_calc_dir) / 'CHGCAR*')))
        if chgcars:
            files_to_transfer['CHGCAR'] = str(chgcars[-1])
    if getattr(self, 'copy_wavecar', False):
        for fname in ('WAVECAR', 'WAVEDER', 'WFULL'):
            wavecar_files = sorted(glob(str(Path(prev_calc_dir) / (fname + '*'))))
            if wavecar_files:
                if fname == 'WFULL':
                    for wavecar_file in wavecar_files:
                        fname = Path(wavecar_file).name
                        fname = fname.split('.')[0]
                        files_to_transfer[fname] = wavecar_file
                else:
                    files_to_transfer[fname] = str(wavecar_files[-1])
    self.files_to_transfer.update(files_to_transfer)
    return self