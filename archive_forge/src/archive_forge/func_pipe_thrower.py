import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
def pipe_thrower():
    raise OSError(errno.EPIPE, 'Bogus pipe error')