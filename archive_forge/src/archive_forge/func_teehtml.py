from __future__ import absolute_import, print_function, division
import io
from petl.compat import text_type, numeric_types, next, PY2, izip_longest, \
from petl.errors import ArgumentError
from petl.util.base import Table, Record
from petl.io.base import getcodec
from petl.io.sources import write_source_from_arg
def teehtml(table, source=None, encoding=None, errors='strict', caption=None, vrepr=text_type, lineterminator='\n', index_header=False, tr_style=None, td_styles=None, truncate=None):
    """
    Return a table that writes rows to a Unicode HTML file as they are
    iterated over.

    """
    source = write_source_from_arg(source)
    return TeeHTMLView(table, source=source, encoding=encoding, errors=errors, caption=caption, vrepr=vrepr, lineterminator=lineterminator, index_header=index_header, tr_style=tr_style, td_styles=td_styles, truncate=truncate)