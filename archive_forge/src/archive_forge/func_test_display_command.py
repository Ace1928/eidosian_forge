import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
def test_display_command(self):
    """EPIPE message is selectively suppressed"""

    def pipe_thrower():
        raise OSError(errno.EPIPE, 'Bogus pipe error')
    self.assertRaises(IOError, pipe_thrower)

    @display_command
    def non_thrower():
        pipe_thrower()
    non_thrower()

    @display_command
    def other_thrower():
        raise OSError(errno.ESPIPE, 'Bogus pipe error')
    self.assertRaises(IOError, other_thrower)