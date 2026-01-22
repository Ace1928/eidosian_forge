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
def scan_directory(self, dirname, exclude_exts=(), exclude_fnames=()):
    """
        Analyze the files contained in directory dirname.

        Args:
            dirname: directory path
            exclude_exts: list of file extensions that should be skipped.
            exclude_fnames: list of file names that should be skipped.

        Returns:
            List of pseudopotential objects.
        """
    for i, ext in enumerate(exclude_exts):
        if not ext.strip().startswith('.'):
            exclude_exts[i] = '.' + ext.strip()
    paths = []
    for fname in os.listdir(dirname):
        _root, ext = os.path.splitext(fname)
        path = os.path.join(dirname, fname)
        if ext in exclude_exts or fname in exclude_fnames or fname.startswith('.') or (not os.path.isfile(path)):
            continue
        paths.append(path)
    pseudos = []
    for path in paths:
        try:
            pseudo = self.parse(path)
        except Exception:
            pseudo = None
        if pseudo is not None:
            pseudos.append(pseudo)
            self._parsed_paths.extend(path)
        else:
            self._wrong_paths.extend(path)
    return pseudos