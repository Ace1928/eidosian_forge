import re
import sys
import six
from genshi.builder import Element
from genshi.core import Stream, Attrs, QName, TEXT, START, END, _ensure, Markup
from genshi.path import Path
def cut(self, buffer, accumulate=False):
    """Copy selection into buffer and remove the selection from the stream.

        >>> from genshi.builder import tag
        >>> buffer = StreamBuffer()
        >>> html = HTML('<html><head><title>Some Title</title></head>'
        ...             '<body>Some <em>body</em> text.</body></html>',
        ...             encoding='utf-8')
        >>> print(html | Transformer('.//em/text()').cut(buffer)
        ...     .end().select('.//em').after(tag.h1(buffer)))
        <html><head><title>Some Title</title></head><body>Some
        <em/><h1>body</h1> text.</body></html>

        Specifying accumulate=True, appends all selected intervals onto the
        buffer. Combining this with the .buffer() operation allows us operate
        on all copied events rather than per-segment. See the documentation on
        buffer() for more information.

        :param buffer: the `StreamBuffer` in which the selection should be
                       stored
        :rtype: `Transformer`
        :note: this transformation will buffer the entire input stream
        """
    return self.apply(CutTransformation(buffer, accumulate))