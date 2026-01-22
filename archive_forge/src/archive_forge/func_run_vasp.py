from __future__ import annotations
import codecs
import contextlib
import hashlib
import itertools
import json
import logging
import math
import os
import re
import subprocess
import warnings
from collections import namedtuple
from enum import Enum, unique
from glob import glob
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal, cast
import numpy as np
import scipy.constants as const
from monty.io import zopen
from monty.json import MontyDecoder, MSONable
from monty.os import cd
from monty.os.path import zpath
from monty.serialization import dumpfn, loadfn
from tabulate import tabulate
from pymatgen.core import SETTINGS, Element, Lattice, Structure, get_el_sp
from pymatgen.electronic_structure.core import Magmom
from pymatgen.util.io_utils import clean_lines
from pymatgen.util.string import str_delimited
def run_vasp(self, run_dir: PathLike='.', vasp_cmd: list | None=None, output_file: PathLike='vasp.out', err_file: PathLike='vasp.err') -> None:
    """
        Write input files and run VASP.

        Args:
            run_dir: Where to write input files and do the run.
            vasp_cmd: Args to be supplied to run VASP. Otherwise, the
                PMG_VASP_EXE in .pmgrc.yaml is used.
            output_file: File to write output.
            err_file: File to write err.
        """
    self.write_input(output_dir=run_dir)
    vasp_cmd = vasp_cmd or SETTINGS.get('PMG_VASP_EXE')
    if not vasp_cmd:
        raise ValueError('No VASP executable specified!')
    vasp_cmd = [os.path.expanduser(os.path.expandvars(t)) for t in vasp_cmd]
    if not vasp_cmd:
        raise RuntimeError('You need to supply vasp_cmd or set the PMG_VASP_EXE in .pmgrc.yaml to run VASP.')
    with cd(run_dir), open(output_file, mode='w', encoding='utf-8') as stdout_file, open(err_file, mode='w', encoding='utf-8', buffering=1) as stderr_file:
        subprocess.check_call(vasp_cmd, stdout=stdout_file, stderr=stderr_file)