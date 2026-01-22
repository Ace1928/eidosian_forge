import re
import sys
import six
from genshi.builder import Element
from genshi.core import Stream, Attrs, QName, TEXT, START, END, _ensure, Markup
from genshi.path import Path
def substitute(self, pattern, replace, count=1):
    """Replace text matching a regular expression.

        Refer to the documentation for ``re.sub()`` for details.

        >>> html = HTML('<html><body>Some text, some more text and '
        ...             '<b>some bold text</b>\\n'
        ...             '<i>some italicised text</i></body></html>',
        ...             encoding='utf-8')
        >>> print(html | Transformer('body/b').substitute('(?i)some', 'SOME'))
        <html><body>Some text, some more text and <b>SOME bold text</b>
        <i>some italicised text</i></body></html>
        >>> tags = tag.html(tag.body('Some text, some more text and\\n',
        ...      Markup('<b>some bold text</b>')))
        >>> print(tags.generate() | Transformer('body').substitute(
        ...     '(?i)some', 'SOME'))
        <html><body>SOME text, some more text and
        <b>SOME bold text</b></body></html>

        :param pattern: A regular expression object or string.
        :param replace: Replacement pattern.
        :param count: Number of replacements to make in each text fragment.
        :rtype: `Transformer`
        """
    return self.apply(SubstituteTransformation(pattern, replace, count))