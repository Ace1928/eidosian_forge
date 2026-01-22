from logging import error
import io
import os
from pprint import pformat
import sys
from warnings import warn
from traitlets.utils.importstring import import_item
from IPython.core import magic_arguments, page
from IPython.core.error import UsageError
from IPython.core.magic import Magics, magics_class, line_magic, magic_escapes
from IPython.utils.text import format_screen, dedent, indent
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.ipstruct import Struct
@line_magic
def colors(self, parameter_s=''):
    """Switch color scheme for prompts, info system and exception handlers.

        Currently implemented schemes: NoColor, Linux, LightBG.

        Color scheme names are not case-sensitive.

        Examples
        --------
        To get a plain black and white terminal::

          %colors nocolor
        """

    def color_switch_err(name):
        warn('Error changing %s color schemes.\n%s' % (name, sys.exc_info()[1]), stacklevel=2)
    new_scheme = parameter_s.strip()
    if not new_scheme:
        raise UsageError("%colors: you must specify a color scheme. See '%colors?'")
    shell = self.shell
    try:
        shell.colors = new_scheme
        shell.refresh_style()
    except:
        color_switch_err('shell')
    try:
        shell.InteractiveTB.set_colors(scheme=new_scheme)
        shell.SyntaxTB.set_colors(scheme=new_scheme)
    except:
        color_switch_err('exception')
    if shell.color_info:
        try:
            shell.inspector.set_active_scheme(new_scheme)
        except:
            color_switch_err('object inspector')
    else:
        shell.inspector.set_active_scheme('NoColor')