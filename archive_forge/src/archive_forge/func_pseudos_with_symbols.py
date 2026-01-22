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
def pseudos_with_symbols(self, symbols):
    """
        Return the pseudos with the given chemical symbols.

        Raises:
            ValueError if one of the symbols is not found or multiple occurrences are present.
        """
    pseudos = self.select_symbols(symbols, ret_list=True)
    found_symbols = [p.symbol for p in pseudos]
    duplicated_elements = [s for s, o in collections.Counter(found_symbols).items() if o > 1]
    if duplicated_elements:
        raise ValueError(f'Found multiple occurrences of symbol(s) {', '.join(duplicated_elements)}')
    missing_symbols = [s for s in symbols if s not in found_symbols]
    if missing_symbols:
        raise ValueError(f'Missing data for symbol(s) {', '.join(missing_symbols)}')
    return pseudos