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
def pprint(self, parameter_s=''):
    """Toggle pretty printing on/off."""
    ptformatter = self.shell.display_formatter.formatters['text/plain']
    ptformatter.pprint = bool(1 - ptformatter.pprint)
    print('Pretty printing has been turned', ['OFF', 'ON'][ptformatter.pprint])