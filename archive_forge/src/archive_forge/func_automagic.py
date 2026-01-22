from IPython.core.magic import Bunch, Magics, magics_class, line_magic
from IPython.testing.skipdoctest import skip_doctest
from logging import error
@line_magic
def automagic(self, parameter_s=''):
    """Make magic functions callable without having to type the initial %.

        Without arguments toggles on/off (when off, you must call it as
        %automagic, of course).  With arguments it sets the value, and you can
        use any of (case insensitive):

         - on, 1, True: to activate

         - off, 0, False: to deactivate.

        Note that magic functions have lowest priority, so if there's a
        variable whose name collides with that of a magic fn, automagic won't
        work for that function (you get the variable instead). However, if you
        delete the variable (del var), the previously shadowed magic function
        becomes visible to automagic again."""
    arg = parameter_s.lower()
    mman = self.shell.magics_manager
    if arg in ('on', '1', 'true'):
        val = True
    elif arg in ('off', '0', 'false'):
        val = False
    else:
        val = not mman.auto_magic
    mman.auto_magic = val
    print('\n' + self.shell.magics_manager.auto_status())