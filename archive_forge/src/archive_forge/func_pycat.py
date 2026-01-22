import io
import os
import pathlib
import re
import sys
from pprint import pformat
from IPython.core import magic_arguments
from IPython.core import oinspect
from IPython.core import page
from IPython.core.alias import AliasError, Alias
from IPython.core.error import UsageError
from IPython.core.magic import  (
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.openpy import source_to_unicode
from IPython.utils.process import abbrev_cwd
from IPython.utils.terminal import set_term_title
from traitlets import Bool
from warnings import warn
@line_magic
def pycat(self, parameter_s=''):
    """Show a syntax-highlighted file through a pager.

        This magic is similar to the cat utility, but it will assume the file
        to be Python source and will show it with syntax highlighting.

        This magic command can either take a local filename, an url,
        an history range (see %history) or a macro as argument.

        If no parameter is given, prints out history of current session up to
        this point. ::

        %pycat myscript.py
        %pycat 7-27
        %pycat myMacro
        %pycat http://www.example.com/myscript.py
        """
    try:
        cont = self.shell.find_user_code(parameter_s, skip_encoding_cookie=False)
    except (ValueError, IOError):
        print('Error: no such file, variable, URL, history range or macro')
        return
    page.page(self.shell.pycolorize(source_to_unicode(cont)))