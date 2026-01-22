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
def get_vasprun_outcar(path: str | Path, parse_dos: bool=True, parse_eigen: bool=True) -> tuple[Vasprun, Outcar]:
    """
    Get a Vasprun and Outcar from a directory.

    Args:
        path: Path to get the vasprun.xml and OUTCAR.
        parse_dos: Whether to parse dos. Defaults to True.
        parse_eigen: Whether to parse eigenvalue. Defaults to True.

    Returns:
        Vasprun and Outcar files.
    """
    path = Path(path)
    vruns = list(glob(str(path / 'vasprun.xml*')))
    outcars = list(glob(str(path / 'OUTCAR*')))
    if len(vruns) == 0 or len(outcars) == 0:
        raise ValueError(f'Unable to get vasprun.xml/OUTCAR from prev calculation in {path}')
    vsfile_fullpath = str(path / 'vasprun.xml')
    outcarfile_fullpath = str(path / 'OUTCAR.gz')
    vsfile = vsfile_fullpath if vsfile_fullpath in vruns else sorted(vruns)[-1]
    outcarfile = outcarfile_fullpath if outcarfile_fullpath in outcars else sorted(outcars)[-1]
    return (Vasprun(vsfile, parse_dos=parse_dos, parse_eigen=parse_eigen), Outcar(outcarfile))