from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, InitVar
from functools import lru_cache
from itertools import chain
from pathlib import Path
import copy
import enum
import json
import os
import pickle
import re
import shlex
import shutil
import typing as T
import hashlib
from .. import build
from .. import dependencies
from .. import programs
from .. import mesonlib
from .. import mlog
from ..compilers import LANGUAGES_USING_LDFLAGS, detect
from ..mesonlib import (
def generate_unity_files(self, target: build.BuildTarget, unity_src: str) -> T.List[mesonlib.File]:
    abs_files: T.List[str] = []
    result: T.List[mesonlib.File] = []
    compsrcs = classify_unity_sources(target.compilers.values(), unity_src)
    unity_size = target.get_option(OptionKey('unity_size'))
    assert isinstance(unity_size, int), 'for mypy'

    def init_language_file(suffix: str, unity_file_number: int) -> T.TextIO:
        unity_src = self.get_unity_source_file(target, suffix, unity_file_number)
        outfileabs = unity_src.absolute_path(self.environment.get_source_dir(), self.environment.get_build_dir())
        outfileabs_tmp = outfileabs + '.tmp'
        abs_files.append(outfileabs)
        outfileabs_tmp_dir = os.path.dirname(outfileabs_tmp)
        if not os.path.exists(outfileabs_tmp_dir):
            os.makedirs(outfileabs_tmp_dir)
        result.append(unity_src)
        return open(outfileabs_tmp, 'w', encoding='utf-8')
    for comp, srcs in compsrcs.items():
        files_in_current = unity_size + 1
        unity_file_number = 0
        ofile = None
        for src in srcs:
            if files_in_current >= unity_size:
                if ofile:
                    ofile.close()
                ofile = init_language_file(comp.get_default_suffix(), unity_file_number)
                unity_file_number += 1
                files_in_current = 0
            ofile.write(f'#include<{src}>\n')
            files_in_current += 1
        if ofile:
            ofile.close()
    for x in abs_files:
        mesonlib.replace_if_different(x, x + '.tmp')
    return result