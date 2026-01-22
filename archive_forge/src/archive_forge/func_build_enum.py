from __future__ import annotations
import os
import shutil
import subprocess
from glob import glob
from typing import TYPE_CHECKING, Literal
from urllib.request import urlretrieve
from monty.json import jsanitize
from monty.serialization import dumpfn, loadfn
from ruamel import yaml
from pymatgen.core import OLD_SETTINGS_FILE, SETTINGS_FILE, Element
from pymatgen.io.cp2k.inputs import GaussianTypeOrbitalBasisSet, GthPotential
from pymatgen.io.cp2k.utils import chunk
def build_enum(fortran_command: str='gfortran') -> bool:
    """Build enum.

    Args:
        fortran_command: The Fortran compiler command.
    """
    cwd = os.getcwd()
    state = True
    try:
        subprocess.call(['git', 'clone', '--recursive', 'https://github.com/msg-byu/enumlib'])
        os.chdir(f'{cwd}/enumlib/symlib/src')
        os.environ['F90'] = fortran_command
        subprocess.call(['make'])
        enum_path = f'{cwd}/enumlib/src'
        os.chdir(enum_path)
        subprocess.call(['make'])
        subprocess.call(['make', 'enum.x'])
        shutil.copy('enum.x', os.path.join('..', '..'))
    except Exception as exc:
        print(exc)
        state = False
    finally:
        os.chdir(cwd)
        shutil.rmtree('enumlib')
    return state