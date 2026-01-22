from __future__ import absolute_import, print_function, division
import io
from petl.compat import next, PY2, text_type
from petl.util.base import Table, asdict
from petl.io.base import getcodec
from petl.io.sources import read_source_from_arg, write_source_from_arg
def fromtext(source=None, encoding=None, errors='strict', strip=None, header=('lines',)):
    """
    Extract a table from lines in the given text file. E.g.::

        >>> import petl as etl
        >>> # setup example file
        ... text = 'a,1\\nb,2\\nc,2\\n'
        >>> with open('example.txt', 'w') as f:
        ...     f.write(text)
        ...
        12
        >>> table1 = etl.fromtext('example.txt')
        >>> table1
        +-------+
        | lines |
        +=======+
        | 'a,1' |
        +-------+
        | 'b,2' |
        +-------+
        | 'c,2' |
        +-------+

        >>> # post-process, e.g., with capture()
        ... table2 = table1.capture('lines', '(.*),(.*)$', ['foo', 'bar'])
        >>> table2
        +-----+-----+
        | foo | bar |
        +=====+=====+
        | 'a' | '1' |
        +-----+-----+
        | 'b' | '2' |
        +-----+-----+
        | 'c' | '2' |
        +-----+-----+

    Note that the strip() function is called on each line, which by default
    will remove leading and trailing whitespace, including the end-of-line
    character - use the `strip` keyword argument to specify alternative
    characters to strip. Set the `strip` argument to `False` to disable this
    behaviour and leave line endings in place.

    """
    source = read_source_from_arg(source)
    return TextView(source, header=header, encoding=encoding, errors=errors, strip=strip)