from __future__ import annotations
import typing as T
import os, sys
from .. import mesonlib
from .. import mlog
from ..mesonlib import Popen_safe
import argparse
def dummy_syms(outfilename: str) -> None:
    """Just touch it so relinking happens always."""
    with open(outfilename, 'w', encoding='utf-8'):
        pass