from __future__ import annotations
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field, InitVar
from functools import lru_cache
import abc
import hashlib
import itertools, pathlib
import os
import pickle
import re
import textwrap
import typing as T
from . import coredata
from . import dependencies
from . import mlog
from . import programs
from .mesonlib import (
from .compilers import (
from .interpreterbase import FeatureNew, FeatureDeprecated
def get_clink_dynamic_linker_and_stdlibs(self) -> T.Tuple['Compiler', T.List[str]]:
    """
        We use the order of languages in `clink_langs` to determine which
        linker to use in case the target has sources compiled with multiple
        compilers. All languages other than those in this list have their own
        linker.
        Note that Vala outputs C code, so Vala sources can use any linker
        that can link compiled C. We don't actually need to add an exception
        for Vala here because of that.
        """
    if self.link_language:
        comp = self.all_compilers[self.link_language]
        return (comp, comp.language_stdlib_only_link_flags(self.environment))
    all_compilers = self.environment.coredata.compilers[self.for_machine]
    dep_langs = self.get_langs_used_by_deps()
    for l in clink_langs:
        if l in self.compilers or l in dep_langs:
            try:
                linker = all_compilers[l]
            except KeyError:
                raise MesonException(f'Could not get a dynamic linker for build target {self.name!r}. Requires a linker for language "{l}", but that is not a project language.')
            stdlib_args: T.List[str] = self.get_used_stdlib_args(linker.language)
            return (linker, stdlib_args)
    for l in clink_langs:
        try:
            comp = self.all_compilers[l]
            return (comp, comp.language_stdlib_only_link_flags(self.environment))
        except KeyError:
            pass
    raise AssertionError(f'Could not get a dynamic linker for build target {self.name!r}')