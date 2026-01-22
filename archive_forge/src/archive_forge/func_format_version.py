import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def format_version(self):
    import warnings
    warnings.warn('The format_version method is deprecated -- the "version" argument to ArgumentParser is no longer supported.', DeprecationWarning)
    formatter = self._get_formatter()
    formatter.add_text(self.version)
    return formatter.format_help()