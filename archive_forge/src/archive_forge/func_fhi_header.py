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
def fhi_header(filename, ppdesc):
    """
        Parse the FHI abinit header. Example:

        Troullier-Martins psp for element  Sc        Thu Oct 27 17:33:22 EDT 1994
            21.00000   3.00000    940714                zatom, zion, pspdat
            1    1    2    0      2001    .00000      pspcod,pspxc,lmax,lloc,mmax,r2well
            1.80626423934776     .22824404341771    1.17378968127746   rchrg,fchrg,qchrg
        """
    lines = _read_nlines(filename, 4)
    try:
        header = _dict_from_lines(lines[:4], [0, 3, 6, 3])
    except ValueError:
        header = _dict_from_lines(lines[:3], [0, 3, 6])
    summary = lines[0]
    return NcAbinitHeader(summary, **header)