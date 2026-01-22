from logging import error
import os
import sys
from IPython.core.error import TryNext, UsageError
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.lib.clipboard import ClipboardEmpty
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.text import SList, strip_email_quotes
from IPython.utils import py3compat
def get_pasted_lines(sentinel, l_input=py3compat.input, quiet=False):
    """ Yield pasted lines until the user enters the given sentinel value.
    """
    if not quiet:
        print("Pasting code; enter '%s' alone on the line to stop or use Ctrl-D." % sentinel)
        prompt = ':'
    else:
        prompt = ''
    while True:
        try:
            l = l_input(prompt)
            if l == sentinel:
                return
            else:
                yield l
        except EOFError:
            print('<EOF>')
            return