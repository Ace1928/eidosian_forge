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
def link_whole(self, targets: T.List[BuildTargetTypes], promoted: bool=False) -> None:
    for t in targets:
        if isinstance(t, (CustomTarget, CustomTargetIndex)):
            if not t.is_linkable_target():
                raise InvalidArguments(f'Custom target {t!r} is not linkable.')
            if t.links_dynamically():
                raise InvalidArguments('Can only link_whole custom targets that are static archives.')
        elif not isinstance(t, StaticLibrary):
            raise InvalidArguments(f'{t!r} is not a static library.')
        elif isinstance(self, SharedLibrary) and (not t.pic):
            msg = f"Can't link non-PIC static library {t.name!r} into shared library {self.name!r}. "
            msg += "Use the 'pic' option to static_library to build with PIC."
            raise InvalidArguments(msg)
        self.check_can_link_together(t)
        if isinstance(self, StaticLibrary):
            self._bundle_static_library(t, promoted)
            if self.install:
                for lib in t.get_internal_static_libraries():
                    self._bundle_static_library(lib, True)
        self.link_whole_targets.append(t)