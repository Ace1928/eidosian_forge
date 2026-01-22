from __future__ import unicode_literals
import six
import pybtex.io
from pybtex.plugin import Plugin
def format_protected(self, text):
    """Format a "protected" piece of text.

        In LaTeX backend, it is formatted as a {braced group}.
        Most other backends would just output the text as-is.
        """
    return text