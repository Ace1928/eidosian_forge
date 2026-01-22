from __future__ import absolute_import
import sys
import os
def cycompile(input_file, options=()):
    from ..Compiler import Version, CmdLine, Main
    options, sources = CmdLine.parse_command_line(list(options or ()) + ['--embed', input_file])
    _debug('Using Cython %s to compile %s', Version.version, input_file)
    result = Main.compile(sources, options)
    if result.num_errors > 0:
        sys.exit(1)