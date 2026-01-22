import re
import sys
import six
from genshi.builder import Element
from genshi.core import Stream, Attrs, QName, TEXT, START, END, _ensure, Markup
from genshi.path import Path
def rename(self, name):
    """Rename matching elements.

        >>> html = HTML('<html><body>Some text, some more text and '
        ...             '<b>some bold text</b></body></html>',
        ...             encoding='utf-8')
        >>> print(html | Transformer('body/b').rename('strong'))
        <html><body>Some text, some more text and <strong>some bold text</strong></body></html>
        """
    return self.apply(RenameTransformation(name))