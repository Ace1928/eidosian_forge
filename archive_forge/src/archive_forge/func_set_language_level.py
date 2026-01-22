from __future__ import absolute_import, print_function
import os
import re
import sys
import io
from . import Errors
from .StringEncoding import EncodedString
from .Scanning import PyrexScanner, FileSourceDescriptor
from .Errors import PyrexError, CompileError, error, warning
from .Symtab import ModuleScope
from .. import Utils
from . import Options
from .Options import CompilationOptions, default_options
from .CmdLine import parse_command_line
from .Lexicon import (unicode_start_ch_any, unicode_continuation_ch_any,
def set_language_level(self, level):
    from .Future import print_function, unicode_literals, absolute_import, division, generator_stop
    future_directives = set()
    if level == '3str':
        level = 3
    else:
        level = int(level)
        if level >= 3:
            future_directives.add(unicode_literals)
    if level >= 3:
        future_directives.update([print_function, absolute_import, division, generator_stop])
    self.language_level = level
    self.future_directives = future_directives
    if level >= 3:
        self.modules['builtins'] = self.modules['__builtin__']