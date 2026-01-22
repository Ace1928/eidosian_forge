from __future__ import annotations
import io
import sys
import typing as ty
import warnings
from functools import reduce
from operator import getitem, mul
from os.path import exists, splitext
import numpy as np
from ._compression import COMPRESSED_FILE_LIKES
from .casting import OK_FLOATS, shared_range
from .externals.oset import OrderedSet
def fname_ext_ul_case(fname: str) -> str:
    """`fname` with ext changed to upper / lower case if file exists

    Check for existence of `fname`.  If it does exist, return unmodified.  If
    it doesn't, check for existence of `fname` with case changed from lower to
    upper, or upper to lower.  Return this modified `fname` if it exists.
    Otherwise return `fname` unmodified

    Parameters
    ----------
    fname : str
        filename.

    Returns
    -------
    mod_fname : str
        filename, maybe with extension of opposite case
    """
    if exists(fname):
        return fname
    froot, ext = splitext(fname)
    if ext == ext.lower():
        mod_fname = froot + ext.upper()
        if exists(mod_fname):
            return mod_fname
    elif ext == ext.upper():
        mod_fname = froot + ext.lower()
        if exists(mod_fname):
            return mod_fname
    return fname