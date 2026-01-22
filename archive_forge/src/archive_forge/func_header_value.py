from collections import namedtuple
import re
import textwrap
import warnings
@property
def header_value(self):
    """(``str`` or ``None``) The header value."""
    return self._header_value