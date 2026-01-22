from __future__ import annotations
import enum
import os.path
import string
import typing as T
from .. import coredata
from .. import mlog
from ..mesonlib import (
from .compilers import Compiler
def get_ccbin_args(self, options: 'KeyedOptionDictType') -> T.List[str]:
    key = OptionKey('ccbindir', machine=self.for_machine, lang=self.language)
    ccbindir = options[key].value
    if isinstance(ccbindir, str) and ccbindir != '':
        return [self._shield_nvcc_list_arg('-ccbin=' + ccbindir, False)]
    else:
        return []