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
def intern_ustring(self, value, encoding=None):
    key = (EncodedString, value, encoding)
    try:
        return self._interned[key]
    except KeyError:
        pass
    value = EncodedString(value)
    if encoding:
        value.encoding = encoding
    self._interned[key] = value
    return value