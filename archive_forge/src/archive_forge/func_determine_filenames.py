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
def determine_filenames(self):
    """
        See https://github.com/mesonbuild/meson/pull/417 for details.

        First we determine the filename template (self.filename_tpl), then we
        set the output filename (self.filename).

        The template is needed while creating aliases (self.get_aliases),
        which are needed while generating .so shared libraries for Linux.

        Besides this, there's also the import library name (self.import_filename),
        which is only used on Windows since on that platform the linker uses a
        separate library called the "import library" during linking instead of
        the shared library (DLL).
        """
    prefix = ''
    suffix = ''
    create_debug_file = False
    self.filename_tpl = self.basic_filename_tpl
    import_filename_tpl = None
    if 'cs' in self.compilers:
        prefix = ''
        suffix = 'dll'
        self.filename_tpl = '{0.prefix}{0.name}.{0.suffix}'
        create_debug_file = True
    elif self.environment.machines[self.for_machine].is_windows():
        suffix = 'dll'
        if self.uses_rust():
            prefix = ''
            import_filename_tpl = '{0.prefix}{0.name}.dll.lib'
            create_debug_file = self.environment.coredata.get_option(OptionKey('debug'))
        elif self.get_using_msvc():
            prefix = ''
            import_filename_tpl = '{0.prefix}{0.name}.lib'
            create_debug_file = self.environment.coredata.get_option(OptionKey('debug'))
        else:
            prefix = 'lib'
            import_filename_tpl = '{0.prefix}{0.name}.dll.a'
        if self.soversion:
            self.filename_tpl = '{0.prefix}{0.name}-{0.soversion}.{0.suffix}'
        else:
            self.filename_tpl = '{0.prefix}{0.name}.{0.suffix}'
    elif self.environment.machines[self.for_machine].is_cygwin():
        suffix = 'dll'
        prefix = 'cyg'
        import_prefix = self.prefix if self.prefix is not None else 'lib'
        import_filename_tpl = import_prefix + '{0.name}.dll.a'
        if self.soversion:
            self.filename_tpl = '{0.prefix}{0.name}-{0.soversion}.{0.suffix}'
        else:
            self.filename_tpl = '{0.prefix}{0.name}.{0.suffix}'
    elif self.environment.machines[self.for_machine].is_darwin():
        prefix = 'lib'
        suffix = 'dylib'
        if self.soversion:
            self.filename_tpl = '{0.prefix}{0.name}.{0.soversion}.{0.suffix}'
        else:
            self.filename_tpl = '{0.prefix}{0.name}.{0.suffix}'
    elif self.environment.machines[self.for_machine].is_android():
        prefix = 'lib'
        suffix = 'so'
        self.filename_tpl = '{0.prefix}{0.name}.{0.suffix}'
    else:
        prefix = 'lib'
        suffix = 'so'
        if self.ltversion:
            self.filename_tpl = '{0.prefix}{0.name}.{0.suffix}.{0.ltversion}'
        elif self.soversion:
            self.filename_tpl = '{0.prefix}{0.name}.{0.suffix}.{0.soversion}'
        else:
            self.filename_tpl = '{0.prefix}{0.name}.{0.suffix}'
    if self.prefix is None:
        self.prefix = prefix
    if self.suffix is None:
        self.suffix = suffix
    self.filename = self.filename_tpl.format(self)
    if import_filename_tpl:
        self.import_filename = import_filename_tpl.format(self)
    self.outputs[0] = self.filename
    if create_debug_file:
        self.debug_filename = os.path.splitext(self.filename)[0] + '.pdb'