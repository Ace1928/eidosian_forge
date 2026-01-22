import sys
import warnings
from IPython.core import ultratb, compilerop
from IPython.core import magic_arguments
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.core.interactiveshell import DummyMod, InteractiveShell
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from IPython.terminal.ipapp import load_default_config
from traitlets import Bool, CBool, Unicode
from IPython.utils.io import ask_yes_no
from typing import Set
@line_magic
@magic_arguments.magic_arguments()
@magic_arguments.argument('-i', '--instance', action='store_true', help='Kill instance instead of call location')
@magic_arguments.argument('-x', '--exit', action='store_true', help='Also exit the current session')
@magic_arguments.argument('-y', '--yes', action='store_true', help='Do not ask confirmation')
def kill_embedded(self, parameter_s=''):
    """%kill_embedded : deactivate for good the current embedded IPython

        This function (after asking for confirmation) sets an internal flag so
        that an embedded IPython will never activate again for the given call
        location. This is useful to permanently disable a shell that is being
        called inside a loop: once you've figured out what you needed from it,
        you may then kill it and the program will then continue to run without
        the interactive shell interfering again.

        Kill Instance Option:

            If for some reasons you need to kill the location where the instance
            is created and not called, for example if you create a single
            instance in one place and debug in many locations, you can use the
            ``--instance`` option to kill this specific instance. Like for the
            ``call location`` killing an "instance" should work even if it is
            recreated within a loop.

        .. note::

            This was the default behavior before IPython 5.2

        """
    args = magic_arguments.parse_argstring(self.kill_embedded, parameter_s)
    print(args)
    if args.instance:
        if not args.yes:
            kill = ask_yes_no('Are you sure you want to kill this embedded instance? [y/N] ', 'n')
        else:
            kill = True
        if kill:
            self.shell._disable_init_location()
            print('This embedded IPython instance will not reactivate anymore once you exit.')
    else:
        if not args.yes:
            kill = ask_yes_no('Are you sure you want to kill this embedded call_location? [y/N] ', 'n')
        else:
            kill = True
        if kill:
            self.shell.embedded_active = False
            print('This embedded IPython  call location will not reactivate anymore once you exit.')
    if args.exit:
        self.shell.ask_exit()