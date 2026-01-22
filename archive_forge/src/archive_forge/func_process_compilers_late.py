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
def process_compilers_late(self) -> None:
    """Processes additional compilers after kwargs have been evaluated.

        This can add extra compilers that might be required by keyword
        arguments, such as link_with or dependencies. It will also try to guess
        which compiler to use if one hasn't been selected already.
        """
    for lang in self.missing_languages:
        self.compilers[lang] = self.all_compilers[lang]
    link_langs = [self.link_language] if self.link_language else clink_langs
    if self.link_targets or self.link_whole_targets:
        for t in itertools.chain(self.link_targets, self.link_whole_targets):
            if isinstance(t, (CustomTarget, CustomTargetIndex)):
                continue
            for name, compiler in t.compilers.items():
                if name in link_langs and name not in self.compilers:
                    self.compilers[name] = compiler
    if not self.compilers:
        for lang in link_langs:
            if lang in self.all_compilers:
                self.compilers[lang] = self.all_compilers[lang]
                break
    self.compilers = OrderedDict(sorted(self.compilers.items(), key=lambda t: sort_clink(t[0])))
    self.post_init()