from pecan.commands import BaseCommand
from warnings import warn
import sys
def invoke_shell(self, locs, banner):
    """
        Invokes the appropriate flavor of the python shell.
        Falls back on the native python shell if the requested
        flavor (ipython, bpython,etc) is not installed.
        """
    shell = self.SHELLS[self.args.shell]
    try:
        shell().invoke(locs, banner)
    except ImportError as e:
        warn('%s is not installed, `%s`, falling back to native shell' % (self.args.shell, e), RuntimeWarning)
        if shell == NativePythonShell:
            raise
        NativePythonShell().invoke(locs, banner)