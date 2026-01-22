import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def print_help(self, file=None):
    if file is None:
        file = _sys.stdout
    self._print_message(self.format_help(), file)