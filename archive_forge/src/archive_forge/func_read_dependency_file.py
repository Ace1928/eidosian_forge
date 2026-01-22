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
def read_dependency_file(self, source_path):
    dep_path = Utils.replace_suffix(source_path, '.dep')
    if os.path.exists(dep_path):
        with open(dep_path, 'rU') as f:
            chunks = [line.split(' ', 1) for line in (l.strip() for l in f) if ' ' in line]
        return chunks
    else:
        return ()