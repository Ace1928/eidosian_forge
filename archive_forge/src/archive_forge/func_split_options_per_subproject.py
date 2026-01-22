from __future__ import annotations
import itertools
import shutil
import os
import textwrap
import typing as T
import collections
from . import build
from . import coredata
from . import environment
from . import mesonlib
from . import mintro
from . import mlog
from .ast import AstIDGenerator, IntrospectionInterpreter
from .mesonlib import MachineChoice, OptionKey
def split_options_per_subproject(self, options: 'coredata.KeyedOptionDictType') -> T.Dict[str, 'coredata.MutableKeyedOptionDictType']:
    result: T.Dict[str, 'coredata.MutableKeyedOptionDictType'] = {}
    for k, o in options.items():
        if k.subproject:
            self.all_subprojects.add(k.subproject)
        result.setdefault(k.subproject, {})[k] = o
    return result