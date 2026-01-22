from __future__ import annotations
import os
import shutil
import subprocess
import warnings
from datetime import datetime
from glob import glob
from pathlib import Path
from shutil import which
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.dev import deprecated
from monty.shutil import decompress_file
from monty.tempfile import ScratchDir
from pymatgen.io.common import VolumetricData
from pymatgen.io.vasp.inputs import Potcar
from pymatgen.io.vasp.outputs import Chgcar
def bader_analysis_from_path(path: str, suffix: str=''):
    """Convenience method to run Bader analysis on a folder containing
    typical VASP output files.

    This method will:

    1. Look for files CHGCAR, AECCAR0, AECCAR2, POTCAR or their gzipped
    counterparts.
    2. If AECCAR* files are present, constructs a temporary reference
    file as AECCAR0 + AECCAR2
    3. Runs Bader analysis twice: once for charge, and a second time
    for the charge difference (magnetization density).

    Args:
        path: path to folder to search in
        suffix: specific suffix to look for (e.g. '.relax1' for 'CHGCAR.relax1.gz'

    Returns:
        summary dict
    """

    def _get_filepath(filename: str, msg: str='') -> str | None:
        paths = glob((glob_pattern := f'{path}/{filename}{suffix}*'))
        if len(paths) == 0:
            warnings.warn(msg or f'no matches for glob_pattern={glob_pattern!r}')
            return None
        if len(paths) > 1:
            paths.sort(reverse=True)
            warnings.warn(f'Multiple files detected, using {os.path.basename(path)}')
        return paths[0]
    chgcar_path = _get_filepath('CHGCAR', 'Could not find CHGCAR!')
    if chgcar_path is not None:
        chgcar = Chgcar.from_file(chgcar_path)
    aeccar0_path = _get_filepath('AECCAR0')
    if not aeccar0_path:
        warnings.warn('Could not find AECCAR0, interpret Bader results with severe caution!')
    aeccar0 = Chgcar.from_file(aeccar0_path) if aeccar0_path else None
    aeccar2_path = _get_filepath('AECCAR2')
    if not aeccar2_path:
        warnings.warn('Could not find AECCAR2, interpret Bader results with severe caution!')
    aeccar2 = Chgcar.from_file(aeccar2_path) if aeccar2_path else None
    potcar_path = _get_filepath('POTCAR')
    if not potcar_path:
        warnings.warn('Could not find POTCAR, cannot calculate charge transfer.')
    potcar = Potcar.from_file(potcar_path) if potcar_path else None
    return bader_analysis_from_objects(chgcar, potcar, aeccar0, aeccar2)