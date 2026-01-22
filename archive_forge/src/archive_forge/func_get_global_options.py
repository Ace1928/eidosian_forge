from __future__ import annotations
import abc
import contextlib, os.path, re
import enum
import itertools
import typing as T
from functools import lru_cache
from .. import coredata
from .. import mlog
from .. import mesonlib
from ..mesonlib import (
from ..arglist import CompilerArgs
def get_global_options(lang: str, comp: T.Type[Compiler], for_machine: MachineChoice, env: 'Environment') -> 'KeyedOptionDictType':
    """Retrieve options that apply to all compilers for a given language."""
    description = f'Extra arguments passed to the {lang}'
    argkey = OptionKey('args', lang=lang, machine=for_machine)
    largkey = argkey.evolve('link_args')
    envkey = argkey.evolve('env_args')
    comp_key = argkey if argkey in env.options else envkey
    comp_options = env.options.get(comp_key, [])
    link_options = env.options.get(largkey, [])
    cargs = coredata.UserArrayOption(description + ' compiler', comp_options, split_args=True, allow_dups=True)
    largs = coredata.UserArrayOption(description + ' linker', link_options, split_args=True, allow_dups=True)
    if comp.INVOKES_LINKER and comp_key == envkey:
        largs.extend_value(comp_options)
    opts: 'KeyedOptionDictType' = {argkey: cargs, largkey: largs}
    return opts