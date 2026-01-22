import os.path
import signal
import sys
import pexpect
def bash(command='bash'):
    """Start a bash shell and return a :class:`REPLWrapper` object."""
    bashrc = os.path.join(os.path.dirname(__file__), 'bashrc.sh')
    return _repl_sh(command, ['--rcfile', bashrc], non_printable_insert='\\[\\]')