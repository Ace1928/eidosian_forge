from __future__ import unicode_literals, with_statement
import re
import pybtex.io
from pybtex.errors import report_error
from pybtex.exceptions import PybtexError
from pybtex import py3compat
def handle_bibstyle(self, style):
    if self.style is not None:
        report_error(AuxDataError('illegal, another \\bibstyle command', self.context))
    else:
        self.style = style