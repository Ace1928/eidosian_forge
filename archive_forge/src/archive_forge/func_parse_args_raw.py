from __future__ import absolute_import, print_function
import os
import shutil
import tempfile
from .Dependencies import cythonize, extended_iglob
from ..Utils import is_package_dir
from ..Compiler import Options
def parse_args_raw(parser, args):
    options, unknown = parser.parse_known_args(args)
    sources = options.sources
    for option in unknown:
        if option.startswith('-'):
            parser.error('unknown option ' + option)
        else:
            sources.append(option)
    del options.sources
    return (options, sources)