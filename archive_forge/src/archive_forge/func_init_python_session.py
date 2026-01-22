from sympy.external.gmpy import GROUND_TYPES
from sympy.external.importtools import version_tuple
from sympy.interactive.printing import init_printing
from sympy.utilities.misc import ARCH
from sympy import *
def init_python_session():
    """Construct new Python session. """
    from code import InteractiveConsole

    class SymPyConsole(InteractiveConsole):
        """An interactive console with readline support. """

        def __init__(self):
            ns_locals = {}
            InteractiveConsole.__init__(self, locals=ns_locals)
            try:
                import rlcompleter
                import readline
            except ImportError:
                pass
            else:
                import os
                import atexit
                readline.set_completer(rlcompleter.Completer(ns_locals).complete)
                readline.parse_and_bind('tab: complete')
                if hasattr(readline, 'read_history_file'):
                    history = os.path.expanduser('~/.sympy-history')
                    try:
                        readline.read_history_file(history)
                    except OSError:
                        pass
                    atexit.register(readline.write_history_file, history)
    return SymPyConsole()