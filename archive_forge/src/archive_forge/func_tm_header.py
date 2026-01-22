from __future__ import annotations
import abc
import collections
import hashlib
import logging
import os
import shutil
import sys
import tempfile
import traceback
from collections import defaultdict, namedtuple
from typing import TYPE_CHECKING
from xml.etree import ElementTree as Et
import numpy as np
from monty.collections import AttrDict, Namespace
from monty.functools import lazy_property
from monty.itertools import iterator_from_slice
from monty.json import MontyDecoder, MSONable
from monty.os.path import find_exts
from tabulate import tabulate
from pymatgen.core import Element
from pymatgen.core.xcfunc import XcFunc
from pymatgen.io.core import ParseError
from pymatgen.util.plotting import add_fig_kwargs, get_ax_fig
@staticmethod
def tm_header(filename, ppdesc):
    """
        Parse the TM abinit header. Example:

        Troullier-Martins psp for element Fm         Thu Oct 27 17:28:39 EDT 1994
        100.00000  14.00000    940714                zatom, zion, pspdat
           1    1    3    0      2001    .00000      pspcod,pspxc,lmax,lloc,mmax,r2well
           0   4.085   6.246    0   2.8786493        l,e99.0,e99.9,nproj,rcpsp
           .00000000    .0000000000    .0000000000    .00000000   rms,ekb1,ekb2,epsatm
           1   3.116   4.632    1   3.4291849        l,e99.0,e99.9,nproj,rcpsp
           .00000000    .0000000000    .0000000000    .00000000   rms,ekb1,ekb2,epsatm
           2   4.557   6.308    1   2.1865358        l,e99.0,e99.9,nproj,rcpsp
           .00000000    .0000000000    .0000000000    .00000000   rms,ekb1,ekb2,epsatm
           3  23.251  29.387    1   2.4776730        l,e99.0,e99.9,nproj,rcpsp
           .00000000    .0000000000    .0000000000    .00000000   rms,ekb1,ekb2,epsatm
           3.62474762267880     .07409391739104    3.07937699839200   rchrg,fchrg,qchrg
        """
    lines = _read_nlines(filename, -1)
    header = []
    for lineno, line in enumerate(lines):
        header.append(line)
        if lineno == 2:
            tokens = line.split()
            _pspcod, _pspxc, lmax, _lloc = map(int, tokens[:4])
            _mmax, _r2well = map(float, tokens[4:6])
            lines = lines[3:]
            break
    projectors = {}
    for idx in range(2 * (lmax + 1)):
        line = lines[idx]
        if idx % 2 == 0:
            proj_info = [line]
        if idx % 2 == 1:
            proj_info.append(line)
            d = _dict_from_lines(proj_info, [5, 4])
            projectors[int(d['l'])] = d
    header.append(lines[idx + 1])
    summary = header[0]
    header = _dict_from_lines(header, [0, 3, 6, 3])
    return NcAbinitHeader(summary, **header)