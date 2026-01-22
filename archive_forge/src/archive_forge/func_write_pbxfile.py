from __future__ import annotations
import functools, uuid, os, operator
import typing as T
from . import backends
from .. import build
from .. import mesonlib
from .. import mlog
from ..mesonlib import MesonBugException, MesonException, OptionKey
def write_pbxfile(self, top_level_dict, ofilename) -> None:
    tmpname = ofilename + '.tmp'
    with open(tmpname, 'w', encoding='utf-8') as ofile:
        ofile.write('// !$*UTF8*$!\n')
        top_level_dict.write(ofile, 0)
    os.replace(tmpname, ofilename)