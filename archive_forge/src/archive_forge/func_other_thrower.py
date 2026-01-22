import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
@display_command
def other_thrower():
    raise OSError(errno.ESPIPE, 'Bogus pipe error')