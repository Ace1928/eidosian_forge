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
def get_devenv(self) -> mesonlib.EnvironmentVariables:
    env = mesonlib.EnvironmentVariables()
    extra_paths = set()
    library_paths = set()
    build_machine = self.environment.machines[MachineChoice.BUILD]
    host_machine = self.environment.machines[MachineChoice.HOST]
    need_wine = not build_machine.is_windows() and host_machine.is_windows()
    for t in self.build.get_targets().values():
        in_default_dir = t.should_install() and (not t.get_install_dir()[2])
        if t.for_machine != MachineChoice.HOST or not in_default_dir:
            continue
        tdir = os.path.join(self.environment.get_build_dir(), self.get_target_dir(t))
        if isinstance(t, build.Executable):
            extra_paths.add(tdir)
            if host_machine.is_windows() or host_machine.is_cygwin():
                library_paths.update(self.determine_windows_extra_paths(t, []))
        elif isinstance(t, build.SharedLibrary):
            library_paths.add(tdir)
    if need_wine:
        library_paths.update(extra_paths)
    if library_paths:
        if need_wine:
            env.prepend('WINEPATH', list(library_paths), separator=';')
        elif host_machine.is_windows() or host_machine.is_cygwin():
            extra_paths.update(library_paths)
        elif host_machine.is_darwin():
            env.prepend('DYLD_LIBRARY_PATH', list(library_paths))
        else:
            env.prepend('LD_LIBRARY_PATH', list(library_paths))
    if extra_paths:
        env.prepend('PATH', list(extra_paths))
    return env