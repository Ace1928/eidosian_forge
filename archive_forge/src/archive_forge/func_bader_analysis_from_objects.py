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
def bader_analysis_from_objects(chgcar: Chgcar, potcar: Potcar | None=None, aeccar0: Chgcar | None=None, aeccar2: Chgcar | None=None):
    """Convenience method to run Bader analysis from a set
    of pymatgen Chgcar and Potcar objects.

    This method will:

    1. If aeccar objects are present, constructs a temporary reference
    file as AECCAR0 + AECCAR2
    2. Runs Bader analysis twice: once for charge, and a second time
    for the charge difference (magnetization density).

    Args:
        chgcar: Chgcar object
        potcar: (optional) Potcar object
        aeccar0: (optional) Chgcar object from aeccar0 file
        aeccar2: (optional) Chgcar object from aeccar2 file

    Returns:
        summary dict
    """
    orig_dir = os.getcwd()
    try:
        with TemporaryDirectory() as tmp_dir:
            os.chdir(tmp_dir)
            if aeccar0 and aeccar2:
                chgref = aeccar0.linear_add(aeccar2)
                chgref_path = f'{tmp_dir}/CHGCAR_ref'
                chgref.write_file(chgref_path)
            else:
                chgref_path = ''
            chgcar.write_file('CHGCAR')
            chgcar_path = f'{tmp_dir}/CHGCAR'
            if potcar:
                potcar.write_file('POTCAR')
                potcar_path = f'{tmp_dir}/POTCAR'
            else:
                potcar_path = ''
            ba = BaderAnalysis(chgcar_filename=chgcar_path, potcar_filename=potcar_path, chgref_filename=chgref_path)
            summary = {'min_dist': [dct['min_dist'] for dct in ba.data], 'charge': [dct['charge'] for dct in ba.data], 'atomic_volume': [dct['atomic_vol'] for dct in ba.data], 'vacuum_charge': ba.vacuum_charge, 'vacuum_volume': ba.vacuum_volume, 'reference_used': bool(chgref_path), 'bader_version': ba.version}
            if potcar:
                charge_transfer = [ba.get_charge_transfer(idx) for idx in range(len(ba.data))]
                summary['charge_transfer'] = charge_transfer
            if chgcar.is_spin_polarized:
                chgcar.data['total'] = chgcar.data['diff']
                chgcar.is_spin_polarized = False
                chgcar.write_file('CHGCAR_mag')
                chgcar_mag_path = f'{tmp_dir}/CHGCAR_mag'
                ba = BaderAnalysis(chgcar_filename=chgcar_mag_path, potcar_filename=potcar_path, chgref_filename=chgref_path)
                summary['magmom'] = [dct['charge'] for dct in ba.data]
    finally:
        os.chdir(orig_dir)
    return summary