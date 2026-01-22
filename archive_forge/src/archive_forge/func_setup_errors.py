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
def setup_errors(self, options, result):
    Errors.init_thread()
    if options.use_listing_file:
        path = result.listing_file = Utils.replace_suffix(result.main_source_file, '.lis')
    else:
        path = None
    Errors.open_listing_file(path=path, echo_to_stderr=options.errors_to_stderr)