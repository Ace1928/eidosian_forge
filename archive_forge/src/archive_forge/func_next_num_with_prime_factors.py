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
def next_num_with_prime_factors(n: int, max_prime_factor: int, must_inc_2: bool=True) -> int:
    """
    Return the next number greater than or equal to n that only has the desired prime factors.

    Args:
        n (int): Initial guess at the grid density
        max_prime_factor (int): the maximum prime factor
        must_inc_2 (bool): 2 must be a prime factor of the result

    Returns:
        int: first product of the prime_factors that is >= n
    """
    if max_prime_factor < 2:
        raise ValueError('Must choose a maximum prime factor greater than 2')
    prime_factors = primes_less_than(max_prime_factor)
    for new_val in itertools.count(start=n):
        if must_inc_2 and new_val % 2 != 0:
            continue
        cur_val_ = new_val
        for j in prime_factors:
            while cur_val_ % j == 0:
                cur_val_ //= j
        if cur_val_ == 1:
            return new_val
    raise ValueError('No factorable number found, not possible.')