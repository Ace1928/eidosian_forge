from __future__ import annotations
import abc
import copy
import hashlib
import itertools
import os
import re
import textwrap
import typing
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from monty.dev import deprecated
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import Element
from pymatgen.io.cp2k.utils import chunk, postprocessor, preprocessor
from pymatgen.io.vasp.inputs import Kpoints as VaspKpoints
from pymatgen.io.vasp.inputs import KpointsSupportedModes
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
@classmethod
def from_el(cls, el: Element, oxi_state: int=0, spin: int=0) -> Self:
    """Create section from element, oxidation state, and spin."""
    el = el if isinstance(el, Element) else Element(el)

    def f(x):
        return {'s': 0, 'p': 1, 'd': 2, 'f': 4}.get(x)

    def f2(x):
        return {0: 2, 1: 6, 2: 10, 3: 14}.get(x)

    def f3(x):
        return {0: 2, 1: 6, 2: 10, 3: 14}.get(x)
    es = el.electronic_structure
    esv = [(int(_[0]), f(_[1]), int(_[2:])) for _ in es.split('.') if '[' not in _]
    esv.sort(key=lambda x: (x[0], x[1]), reverse=True)
    tmp = oxi_state
    l_alpha = []
    l_beta = []
    nel_alpha = []
    nel_beta = []
    n_alpha = []
    n_beta = []
    unpaired_orbital: tuple[int, int, int] = (0, 0, 0)
    while tmp:
        tmp2 = -min((esv[0][2], tmp)) if tmp > 0 else min((f2(esv[0][1]) - esv[0][2], -tmp))
        l_alpha.append(esv[0][1])
        l_beta.append(esv[0][1])
        nel_alpha.append(tmp2)
        nel_beta.append(tmp2)
        n_alpha.append(esv[0][0])
        n_beta.append(esv[0][0])
        tmp += tmp2
        unpaired_orbital = (esv[0][0], esv[0][1], esv[0][2] + tmp2)
        esv.pop(0)
    if unpaired_orbital is None:
        raise ValueError('unpaired_orbital cannot be None.')
    if spin == 'low-up':
        spin = unpaired_orbital[2] % 2
    elif spin == 'low-down':
        spin = -(unpaired_orbital[2] % 2)
    elif spin == 'high-up':
        spin = unpaired_orbital[2] % (f2(unpaired_orbital[1]) // 2)
    elif spin == 'high-down':
        spin = -(unpaired_orbital[2] % (f2(unpaired_orbital[1]) // 2))
    if spin:
        for i in reversed(range(len(nel_alpha))):
            nel_alpha[i] += min((spin, f3(l_alpha[i]) - oxi_state))
            nel_beta[i] -= min((spin, f3(l_beta[i]) - oxi_state))
            if spin > 0:
                spin -= min((spin, f3(l_alpha[i]) - oxi_state))
            else:
                spin += min((spin, f3(l_beta[i]) - oxi_state))
    return BrokenSymmetry(l_alpha=l_alpha, l_beta=l_beta, nel_alpha=nel_alpha, nel_beta=nel_beta, n_beta=n_beta, n_alpha=n_alpha)