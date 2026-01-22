from __future__ import unicode_literals
import  six
from pybtex import richtext
from pybtex.exceptions import PybtexError
from pybtex.py3compat import fix_unicode_literals_in_doctest
@node
def optional_field(children, data, *args, **kwargs):
    assert not children
    return optional[field(*args, **kwargs)].format_data(data)