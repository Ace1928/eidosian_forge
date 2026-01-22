from __future__ import unicode_literals
import  six
from pybtex import richtext
from pybtex.exceptions import PybtexError
from pybtex.py3compat import fix_unicode_literals_in_doctest
@node
def first_of(children, data):
    """Return first nonempty child."""
    for child in _format_list(children, data):
        if child:
            return child
    return richtext.Text()